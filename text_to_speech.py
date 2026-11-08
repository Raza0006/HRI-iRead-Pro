import threading
import pyttsx3


# ==========================================================
# SETTINGS
# ==========================================================

DEFAULT_RATE = 165
DEFAULT_VOLUME = 1.0


# ==========================================================
# TEXT TO SPEECH CLASS
# ==========================================================

class TextToSpeech:

    def __init__(
        self,
        rate=DEFAULT_RATE,
        volume=DEFAULT_VOLUME
    ):

        self.rate = rate
        self.volume = volume

        self.engine = None

        self.lock = threading.Lock()


    # ======================================================
    # LOAD TTS ENGINE
    # ======================================================

    def _get_engine(self):

        if self.engine is None:

            self.engine = pyttsx3.init()

            self.engine.setProperty(
                "rate",
                self.rate
            )

            self.engine.setProperty(
                "volume",
                self.volume
            )

        return self.engine


    # ======================================================
    # SPEAK TEXT
    # ======================================================

    def speak(self, text):

        if not text:

            return

        # Make sure text is a normal string
        text = str(text).strip()

        if not text:

            return


        # Prevent two speech requests from
        # trying to use the engine simultaneously.
        with self.lock:

            engine = self._get_engine()

            engine.say(text)

            engine.runAndWait()


    # ======================================================
    # CHANGE SPEECH RATE
    # ======================================================

    def set_rate(self, rate):

        self.rate = int(rate)

        if self.engine is not None:

            self.engine.setProperty(
                "rate",
                self.rate
            )


    # ======================================================
    # CHANGE VOLUME
    # ======================================================

    def set_volume(self, volume):

        volume = max(
            0.0,
            min(
                1.0,
                float(volume)
            )
        )

        self.volume = volume

        if self.engine is not None:

            self.engine.setProperty(
                "volume",
                self.volume
            )


# ==========================================================
# GLOBAL TTS INSTANCE
# ==========================================================

_tts = None


def _get_tts():

    global _tts

    if _tts is None:

        _tts = TextToSpeech()

    return _tts


# ==========================================================
# PUBLIC FUNCTION
# ==========================================================

def speak_text(text):
    """
    Speak a supplied text string.

    Example:

        speak_text(
            "This is the detected document text."
        )
    """

    tts = _get_tts()

    tts.speak(text)


# ==========================================================
# OPTIONAL SETTINGS FUNCTIONS
# ==========================================================

def set_speech_rate(rate):

    tts = _get_tts()

    tts.set_rate(rate)


def set_speech_volume(volume):

    tts = _get_tts()

    tts.set_volume(volume)

