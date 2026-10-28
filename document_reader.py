import cv2
import re
import statistics
import numpy as np

from pathlib import Path
from paddleocr import PaddleOCR


# ==========================================================
# SETTINGS
# ==========================================================

CONFIDENCE_THRESHOLD = 0.68

# Maximum image width passed into OCR.
# Larger images are resized while maintaining aspect ratio.
MAX_OCR_WIDTH = 1100


# ==========================================================
# DOCUMENT OCR CLASS
# ==========================================================

class DocumentOCR:

    def __init__(self):

        print("Loading document OCR model...")

        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",

            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print("Document OCR model loaded.")


    # ======================================================
    # IMAGE LOADING
    # ======================================================

    def _load_image(self, image):

        # --------------------------------------------------
        # IMAGE PROVIDED AS FILE PATH
        # --------------------------------------------------

        if isinstance(image, (str, Path)):

            frame = cv2.imread(
                str(image)
            )

            if frame is None:

                raise FileNotFoundError(
                    f"Could not load image: {image}"
                )

            return frame


        # --------------------------------------------------
        # IMAGE PROVIDED DIRECTLY BY YOLO / OPENCV
        # --------------------------------------------------

        if isinstance(image, np.ndarray):

            return image.copy()


        raise TypeError(
            "Document image must be either "
            "a file path or OpenCV numpy array."
        )


    # ======================================================
    # PREPARE IMAGE FOR OCR
    # ======================================================

    def _prepare_image(self, frame):

        height, width = frame.shape[:2]

        if width > MAX_OCR_WIDTH:

            scale = (
                MAX_OCR_WIDTH
                / width
            )

            new_width = MAX_OCR_WIDTH

            new_height = int(
                height * scale
            )

            frame = cv2.resize(
                frame,
                (
                    new_width,
                    new_height
                )
            )

        return frame


    # ======================================================
    # CLEAN OCR TEXT
    # ======================================================

    def _clean_text(self, text):

        text = text.strip()

        # Replace repeated whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        # Remove spaces before punctuation
        text = re.sub(
            r"\s+([,.!?;:%])",
            r"\1",
            text
        )

        # Remove obvious repeated-character OCR garbage
        text = re.sub(
            r"\b(.)\1{5,}\w*\b",
            "",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text


    # ======================================================
    # EXTRACT OCR TEXT AND POSITIONS
    # ======================================================

    def _extract_items(self, results):

        items = []

        for result in results:

            data = result.json["res"]

            texts = data.get(
                "rec_texts",
                []
            )

            scores = data.get(
                "rec_scores",
                []
            )

            boxes = data.get(
                "rec_boxes"
            )

            polygons = data.get(
                "rec_polys"
            )


            for index, (text, score) in enumerate(
                zip(texts, scores)
            ):

                score = float(score)


                # ------------------------------------------
                # IGNORE LOW-CONFIDENCE TEXT
                # ------------------------------------------

                if (
                    score
                    < CONFIDENCE_THRESHOLD
                ):

                    continue


                text = self._clean_text(
                    text
                )


                if not text:

                    continue


                # ------------------------------------------
                # NORMAL RECTANGULAR BOX
                # ------------------------------------------

                if (
                    boxes is not None
                    and index < len(boxes)
                ):

                    box = boxes[index]

                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])


                # ------------------------------------------
                # POLYGON FALLBACK
                # ------------------------------------------

                elif (
                    polygons is not None
                    and index < len(polygons)
                ):

                    polygon = polygons[index]

                    xs = [
                        float(point[0])
                        for point in polygon
                    ]

                    ys = [
                        float(point[1])
                        for point in polygon
                    ]

                    x1 = min(xs)
                    y1 = min(ys)

                    x2 = max(xs)
                    y2 = max(ys)


                else:

                    continue


                height = (
                    y2 - y1
                )


                items.append(
                    {
                        "text": text,
                        "score": score,

                        "x1": x1,
                        "y1": y1,

                        "x2": x2,
                        "y2": y2,

                        "centre_y": (
                            y1 + y2
                        ) / 2,

                        "height": height
                    }
                )


        return items


    # ======================================================
    # JOIN TEXT ON SAME LINE
    # ======================================================

    def _join_line(self, line):

        if not line:

            return ""


        line = sorted(
            line,
            key=lambda item:
            item["x1"]
        )


        text = ""


        for item in line:

            word = self._clean_text(
                item["text"]
            )

            if not word:

                continue


            # First item
            if not text:

                text = word

                continue


            # Punctuation should not receive
            # an extra leading space
            if word.startswith(
                (
                    ".",
                    ",",
                    "!",
                    "?",
                    ";",
                    ":",
                    "%"
                )
            ):

                text += word

            else:

                text += (
                    " " + word
                )


        return self._clean_text(
            text
        )


    # ======================================================
    # RECONSTRUCT DOCUMENT READING ORDER
    # ======================================================

    def _build_document_text(
        self,
        items
    ):

        if not items:

            return ""


        heights = [
            item["height"]
            for item in items
            if item["height"] > 0
        ]


        if heights:

            median_height = (
                statistics.median(
                    heights
                )
            )

        else:

            median_height = 20


        # Text regions close vertically are
        # probably on the same line.
        line_tolerance = max(
            8,
            median_height * 0.55
        )


        # --------------------------------------------------
        # INITIAL TOP → BOTTOM SORT
        # --------------------------------------------------

        items = sorted(
            items,
            key=lambda item: (
                item["centre_y"],
                item["x1"]
            )
        )


        # --------------------------------------------------
        # GROUP INTO TEXT LINES
        # --------------------------------------------------

        lines = []

        current_line = []

        current_y = None


        for item in items:

            if not current_line:

                current_line = [
                    item
                ]

                current_y = (
                    item["centre_y"]
                )

                continue


            if abs(
                item["centre_y"]
                - current_y
            ) <= line_tolerance:

                current_line.append(
                    item
                )

                current_y = sum(
                    word["centre_y"]
                    for word in current_line
                ) / len(
                    current_line
                )


            else:

                lines.append(
                    current_line
                )

                current_line = [
                    item
                ]

                current_y = (
                    item["centre_y"]
                )


        if current_line:

            lines.append(
                current_line
            )


        # --------------------------------------------------
        # PROCESS EACH LINE
        # --------------------------------------------------

        processed_lines = []


        for line in lines:

            line_text = (
                self._join_line(
                    line
                )
            )


            if not line_text:

                continue


            processed_lines.append(
                {
                    "text": line_text,

                    "y1": min(
                        item["y1"]
                        for item in line
                    ),

                    "y2": max(
                        item["y2"]
                        for item in line
                    )
                }
            )


        processed_lines.sort(
            key=lambda line:
            line["y1"]
        )


        # --------------------------------------------------
        # RECONSTRUCT PARAGRAPHS
        # --------------------------------------------------

        paragraphs = []

        current_paragraph = ""

        previous_line = None


        for line in processed_lines:

            text = line["text"]


            if previous_line is None:

                current_paragraph = text


            else:

                vertical_gap = (
                    line["y1"]
                    - previous_line["y2"]
                )


                # Large vertical gap = likely
                # paragraph break
                if (
                    vertical_gap
                    > median_height * 0.9
                ):

                    if current_paragraph:

                        paragraphs.append(
                            current_paragraph
                        )

                    current_paragraph = (
                        text
                    )


                else:

                    # --------------------------------------
                    # HANDLE WORD SPLIT WITH HYPHEN
                    # --------------------------------------

                    if (
                        current_paragraph.endswith(
                            "-"
                        )
                        and text
                        and text[0].islower()
                    ):

                        current_paragraph = (
                            current_paragraph[:-1]
                            + text
                        )

                    else:

                        current_paragraph += (
                            " " + text
                        )


            previous_line = line


        if current_paragraph:

            paragraphs.append(
                current_paragraph
            )


        return "\n\n".join(
            paragraphs
        )


    # ======================================================
    # SCAN DOCUMENT
    # ======================================================

    def scan(self, image):

        frame = self._load_image(
            image
        )

        frame = self._prepare_image(
            frame
        )


        # --------------------------------------------------
        # PADDLE OCR
        # --------------------------------------------------

        results = self.ocr.predict(
            frame
        )


        # --------------------------------------------------
        # EXTRACT RECOGNISED REGIONS
        # --------------------------------------------------

        items = self._extract_items(
            results
        )


        # --------------------------------------------------
        # RECONSTRUCT READING ORDER
        # --------------------------------------------------

        text = self._build_document_text(
            items
        )


        # --------------------------------------------------
        # CALCULATE OVERALL CONFIDENCE
        # --------------------------------------------------

        if items:

            average_confidence = (
                sum(
                    item["score"]
                    for item in items
                )
                / len(items)
            )

        else:

            average_confidence = 0.0


        return {
            "text": text.strip(),

            "confidence": float(
                average_confidence
            ),

            "regions": items
        }


# ==========================================================
# GLOBAL OCR INSTANCE
# ==========================================================

_document_reader = None


def _get_reader():

    global _document_reader

    # Load PaddleOCR only once.
    #
    # The first document will create the model.
    # Every document after that reuses it.
    if _document_reader is None:

        _document_reader = (
            DocumentOCR()
        )

    return _document_reader


# ==========================================================
# PUBLIC FUNCTION
# ==========================================================

def read_document(image):
    """
    Convert a detected/cropped document image into readable text.

    image can be:

    1. An OpenCV / YOLO numpy image:
       read_document(document_crop)

    2. An image file:
       read_document("document.jpg")

    Returns:
        str: reconstructed document text
    """

    reader = _get_reader()

    result = reader.scan(
        image
    )

    return result["text"]


# ==========================================================
# OPTIONAL DETAILED FUNCTION
# ==========================================================

def scan_document(image):
    """
    Same as read_document(), but also returns
    confidence and OCR region information.
    """

    reader = _get_reader()

    return reader.scan(
        image
    )


# ==========================================================
# OPTIONAL PRELOAD
# ==========================================================

def preload_ocr():
    """
    Load PaddleOCR before the first document is detected.

    Useful if the main application wants to avoid
    model-loading delay when the first document appears.
    """

    _get_reader()