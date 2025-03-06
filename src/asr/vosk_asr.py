import sounddevice as sd
import vosk
import json
from .base import SpeechToText


class VoskASR(SpeechToText):
    def __init__(self, model_dir: str):
        """音声認識モデルを初期化

        Args:
            model_dir (str): Voskモデルのディレクトリパス
        """
        super().__init__()
        self.model = vosk.Model(model_dir)

    def audio_input(self) -> str:
        """マイク入力から音声を認識し、テキストを返す

        Returns:
            str: 認識したテキスト
        """
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            rec = vosk.KaldiRecognizer(self.model, 16000)
            while True:
                data = self.q.get()
                if not self.running:
                    continue
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").replace(" ", "")
                    if text:
                        return text
