import io
import soundfile as sf
import numpy as np
from abc import ABC, abstractmethod


class TextToSpeech(ABC):
    def __init__(self, port):
        self.port = port

    @abstractmethod
    def print_speakers(self):
        """使えるキャラクター一覧を表示"""
        pass

    @abstractmethod
    def synthesize(self, text: str, speaker_id: int) -> tuple[np.ndarray, int]:
        """音声合成して再生する

        Args:
            text (str): 音声合成したいテキスト
            speaker_id (int): キャラクターID

        Returns:
            tuple[np.ndarray, int]: 音声データとサンプリングレート
        """
        pass

    @abstractmethod
    async def synthesize_async(
        self, text: str, speaker_id: int
    ) -> tuple[np.ndarray, int]:
        """非同期で音声合成を行う

        Args:
            text (str): 音声合成したいテキスト
            speaker_id (int): キャラクターID

        Returns:
            tuple[np.ndarray, int]: 音声データとサンプリングレート
        """
        pass

    def _read_wav(self, wav_data: bytes) -> tuple[np.ndarray, int]:
        with io.BytesIO(wav_data) as wav_file:
            data, sr = sf.read(wav_file, dtype="float32")
        return data, sr
