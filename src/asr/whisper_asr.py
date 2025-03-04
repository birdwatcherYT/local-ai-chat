import sounddevice as sd
import numpy as np
import webrtcvad
from faster_whisper import WhisperModel
import wave
from collections import deque
from .base import SpeechToText


class WhisperASR(SpeechToText):
    def __init__(
        self,
        model_name: str,
        compute_type: str,
        vad_filter: bool,
        # vad設定
        sensitivity: int,
        hangover_threshold: str,
        pre_buffer_frames: str,
    ):
        """音声認識モデルを初期化

        Args:
            model_name (str): Faster-Whisperモデル名（例: "small", "turbo"）
        """
        super().__init__()
        # VADの初期化
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(sensitivity)  # 感度設定（0〜3、3が最も厳しい）
        # Whisperモデルのロード
        self.model = WhisperModel(model_name, compute_type=compute_type)
        self.buffer = []  # 音声データを蓄積するバッファ
        self.is_speaking = False  # 発話中かどうかのフラグ
        self.sample_rate = 16000  # サンプリングレート
        self.vad_filter = vad_filter
        self.hangover_threshold = hangover_threshold
        self.pre_buffer_frames = pre_buffer_frames

    def save_wav(self, file_name: str, audio_data: bytes):
        """音声データをWAVファイルとして保存

        Args:
            file_name (str): 保存するファイル名
            audio_data (bytes): 音声データ
        """
        with wave.open(file_name, "wb") as wf:
            wf.setnchannels(1)  # モノラル
            wf.setsampwidth(2)  # 16ビット
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)

    def audio_input(self) -> str:
        """マイク入力から音声を認識し、テキストを返す

        Returns:
            str: 認識したテキスト
        """
        # ハングオーバー処理用の無音カウンター
        silence_counter = 0
        # プリバッファ：直前の数フレームを保持
        pre_buffer = deque(maxlen=self.pre_buffer_frames)

        with sd.RawInputStream(
            samplerate=self.sample_rate,  # サンプリングレート: 16kHz
            blocksize=320,  # 20ms分のフレーム（320サンプル）
            dtype="int16",  # データ型: 16ビット整数
            channels=1,  # モノラル
            callback=self._callback,
        ):
            while True:
                # キューから音声データを取得（20ms分のバイトデータ）
                data = self.q.get()
                # まずは常にプリバッファに追加
                pre_buffer.append(data)
                if not self.running:
                    continue
                # VADで音声区間を検出
                is_speech = self.vad.is_speech(data, sample_rate=self.sample_rate)

                if is_speech:
                    # 発話開始直前なら、プリバッファの内容を先にコピーする
                    if not self.is_speaking:
                        self.buffer.extend(pre_buffer)
                        self.is_speaking = True
                    self.buffer.append(data)
                    silence_counter = 0  # 音声が検出されたので無音カウンターをリセット
                elif self.is_speaking:
                    silence_counter += 1
                    self.buffer.append(data)
                    # 一定数の無音フレームが連続した場合、発話終了と判断
                    if silence_counter >= self.hangover_threshold:
                        # バッファ内の音声データを結合
                        audio_bytes = b"".join(self.buffer)

                        # WAVファイルに保存（デバッグ用）
                        # self.save_wav("temp.wav", audio_bytes)

                        # 音声データをfloat32のNumPyアレイに変換
                        audio = (
                            np.frombuffer(audio_bytes, dtype=np.int16).astype(
                                np.float32
                            )
                            / 32768.0
                        )
                        # Whisperで音声を認識
                        segments, _ = self.model.transcribe(
                            audio, language="ja", vad_filter=self.vad_filter
                        )
                        # 認識結果をテキストとして結合
                        text = " ".join([segment.text for segment in segments]).strip()

                        # バッファと状態をリセット
                        self.buffer = []
                        self.is_speaking = False
                        silence_counter = 0

                        if text:
                            return text
