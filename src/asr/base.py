import queue
from abc import ABC, abstractmethod


class SpeechToText(ABC):
    def __init__(self):
        self.running = True
        self.q = queue.Queue()

    def pause(self):
        self.running = False  # 音声入力を無効化

    def resume(self):
        self.running = True  # 音声入力を再開

    def _callback(self, indata, frames, time_info, status):
        """音声データをキューに追加

        Args:
            indata: マイクからの生音声データ
            frames: フレーム数
            time_info: 時間情報
            status: ステータス情報
        """
        if status:
            print(status, flush=True)
        self.q.put(bytes(indata))

    @abstractmethod
    def audio_input(self) -> str:
        pass
