import numpy as np

# vvとインターフェースが共通なのでそのまま利用する
from .voicevox import VoiceVox
from .base import TextToSpeech


# SEE http://localhost:10101/docs


class AivisSpeech(VoiceVox):
    def __init__(self):
        super().__init__()
        self.port = 10101
