# import sounddevice as sd
# import numpy as np
# import webrtcvad
# from faster_whisper import WhisperModel
# from collections import deque
# from .base import SpeechToText


# class WhisperASR(SpeechToText):
#     def __init__(
#         self,
#         model_name: str,
#         compute_type: str,
#         vad_filter: bool,
#         sensitivity: int,
#         hangover_threshold: int,  # int型に修正（文字列ではなく数値として扱う）
#         pre_buffer_frames: int,  # int型に修正
#     ):
#         super().__init__()
#         self.vad = webrtcvad.Vad()
#         self.vad.set_mode(sensitivity)
#         self.model = WhisperModel(model_name, compute_type=compute_type)
#         self.buffer = []  # 音声データを蓄積するバッファ
#         self.is_speaking = False
#         self.sample_rate = 16000
#         self.vad_filter = vad_filter
#         self.hangover_threshold = hangover_threshold
#         self.pre_buffer_frames = pre_buffer_frames
#         self.partial_text = ""  # 部分認識結果を保持
#         self.recognition_interval = 50  # 認識間隔（例: 50フレーム = 1秒）

#     def audio_input(self) -> str:
#         """マイク入力から音声を認識し、テキストを返す

#         Returns:
#             str: 認識したテキスト
#         """
#         silence_counter = 0
#         pre_buffer = deque(maxlen=self.pre_buffer_frames)
#         frame_counter = 0  # 認識間隔を管理するためのカウンター

#         with sd.RawInputStream(
#             samplerate=self.sample_rate,
#             blocksize=320,
#             dtype="int16",
#             channels=1,
#             callback=self._callback,
#         ):
#             while True:
#                 data = self.q.get()
#                 pre_buffer.append(data)
#                 if not self.running:  # runningフラグは基底クラスから継承と仮定
#                     continue

#                 is_speech = self.vad.is_speech(data, sample_rate=self.sample_rate)

#                 if is_speech:
#                     if not self.is_speaking:
#                         self.buffer.extend(pre_buffer)
#                         self.is_speaking = True
#                     self.buffer.append(data)
#                     silence_counter = 0
#                     frame_counter += 1

#                     # 発話中に一定間隔で部分認識を実行
#                     if self.is_speaking and frame_counter >= self.recognition_interval:
#                         audio_bytes = b"".join(self.buffer)
#                         audio = (
#                             np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
#                             / 32768.0
#                         )
#                         segments, _ = self.model.transcribe(
#                             audio, language="ja", vad_filter=self.vad_filter
#                         )
#                         self.partial_text = " ".join([segment.text for segment in segments]).strip()
#                         print("part", self.partial_text, flush=True)
#                         frame_counter = 0  # カウンターをリセット

#                 elif self.is_speaking:
#                     silence_counter += 1
#                     self.buffer.append(data)
#                     if silence_counter >= self.hangover_threshold:
#                         # 発話終了時にバッファ全体を認識
#                         audio_bytes = b"".join(self.buffer)
#                         audio = (
#                             np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
#                             / 32768.0
#                         )
#                         segments, _ = self.model.transcribe(
#                             audio, language="ja", vad_filter=self.vad_filter
#                         )
#                         final_text = " ".join([segment.text for segment in segments]).strip()

#                         # リセット
#                         self.buffer = []
#                         self.is_speaking = False
#                         self.partial_text = ""
#                         silence_counter = 0
#                         frame_counter = 0

#                         if final_text:
#                             return final_text
import sounddevice as sd
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
from collections import deque
from .base import SpeechToText


class WhisperASR(SpeechToText):
    def __init__(
        self,
        model_name: str,
        compute_type: str,
        vad_filter: bool,
        sensitivity: int,
        hangover_threshold: int,
        pre_buffer_frames: int,
    ):
        super().__init__()
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(sensitivity)
        self.model = WhisperModel(model_name, compute_type=compute_type)
        self.buffer = []  # 音声データを蓄積するバッファ
        self.is_speaking = False
        self.sample_rate = 16000
        self.vad_filter = vad_filter
        self.hangover_threshold = hangover_threshold
        self.pre_buffer_frames = pre_buffer_frames
        self.partial_text = ""  # 部分認識結果を保持
        self.segment_size = 50  # セグメントサイズ（例: 50フレーム = 1秒）
        self.overlap_size = 25  # 重なりサイズ（例: 25フレーム = 0.5秒）
        self.processed_frames = 0  # 既に推論済みのフレーム数

    def audio_input(self) -> str:
        """マイク入力から音声を認識し、テキストを返す

        Returns:
            str: 認識したテキスト
        """
        silence_counter = 0
        pre_buffer = deque(maxlen=self.pre_buffer_frames)

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=320,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            while True:
                data = self.q.get()
                pre_buffer.append(data)
                if not self.running:  # runningフラグは基底クラスから継承
                    continue

                is_speech = self.vad.is_speech(data, sample_rate=self.sample_rate)

                if is_speech:
                    if not self.is_speaking:
                        self.buffer.extend(pre_buffer)
                        self.is_speaking = True
                    self.buffer.append(data)
                    silence_counter = 0

                    # バッファがセグメントサイズ以上になったら部分認識
                    if len(self.buffer) >= self.processed_frames + self.segment_size:
                        # 未推論部分＋重なり部分を取得
                        start_idx = max(0, self.processed_frames - self.overlap_size)
                        audio_bytes = b"".join(self.buffer[start_idx:])
                        audio = (
                            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        segments, _ = self.model.transcribe(
                            audio, language="ja", vad_filter=self.vad_filter
                        )
                        new_text = " ".join([segment.text for segment in segments]).strip()

                        # 重なり部分を考慮して部分テキストを更新
                        if self.processed_frames == 0:
                            self.partial_text = new_text
                        else:
                            # 重なり部分を除いた新テキストを追加
                            self.partial_text += " " + new_text
                        print("part", self.partial_text, flush=True)

                        # 推論済みフレーム数を更新
                        self.processed_frames = len(self.buffer)

                elif self.is_speaking:
                    silence_counter += 1
                    self.buffer.append(data)
                    if silence_counter >= self.hangover_threshold:
                        # 発話終了時にバッファ全体を認識
                        audio_bytes = b"".join(self.buffer)
                        audio = (
                            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        segments, _ = self.model.transcribe(
                            audio, language="ja", vad_filter=self.vad_filter
                        )
                        final_text = " ".join([segment.text for segment in segments]).strip()

                        # リセット
                        self.buffer = []
                        self.is_speaking = False
                        self.partial_text = ""
                        self.processed_frames = 0
                        silence_counter = 0

                        if final_text:
                            return final_text
