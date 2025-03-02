import queue
import sounddevice as sd
import vosk
import json


class SpeechRecognizer:
    def __init__(self, model_path: str):
        """音声認識モデルを初期化

        Args:
            model_path (str): Voskモデルのディレクトリパス
        """
        self.model = vosk.Model(model_path)
        self.q = queue.Queue()

    def _callback(self, indata, frames, time, status):
        """音声データをキューに追加"""
        if status:
            print(status, flush=True)
        self.q.put(bytes(indata))

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
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").replace(" ", "")
                    if text:
                        return text
