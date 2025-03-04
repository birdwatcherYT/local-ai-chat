import io
import aiohttp
import asyncio
import requests
import soundfile as sf
import numpy as np
from .base import TextToSpeech

# SEE http://localhost:50032/docs


class CoeiroInk(TextToSpeech):
    def __init__(self):
        super().__init__(50032)

    def print_speakers(self):
        """使えるキャラクター一覧を表示"""
        response = requests.get(f"http://localhost:{self.port}/v1/speakers")

        if response.status_code == 200:
            speakers = response.json()
            print("speakerUuid speakerName styleId styleName")
            for speaker in speakers:
                for style in speaker["styles"]:
                    print(
                        f"{speaker['speakerUuid']} {speaker['speakerName']} {style['styleId']} {style['styleName']}"
                    )
        else:
            print(f"Error: {response.status_code}")

    def synthesize(
        self, text: str, speaker_uuid: str, style_id: int
    ) -> tuple[np.ndarray, int]:
        """音声合成して再生する

        Args:
            text (str):
            speaker_uuid (str): _description_
            style_id (int): スタイルID

        Returns:
            tuple[np.ndarray, int]: 音声データとサンプリングレート
        """
        sr = 24000
        payload = {
            "speakerUuid": speaker_uuid,
            "styleId": style_id,
            "text": text,
            "speedScale": 1.0,
            "volumeScale": 1.0,
            "pitchScale": 0.0,
            "intonationScale": 1.0,
            "prePhonemeLength": 0.1,
            "postPhonemeLength": 0.1,
            "outputSamplingRate": sr,
        }
        response = requests.post(
            f"http://localhost:{self.port}/v1/synthesis", json=payload
        )
        if response.status_code == 200:
            # WAVデータをメモリから読み込む
            data, _sr = self._read_wav(response.content)
            assert _sr == sr
            return data, sr
        else:
            print(f"Error: {response.text}")

    async def synthesize_async(
        self, text: str, speaker_uuid: str, style_id: int
    ) -> tuple[np.ndarray, int]:
        """音声合成して再生する (非同期処理)

        Args:
            text (str): 合成するテキスト
            speaker_uuid (str): キャラクターの UUID
            style_id (int): スタイル ID

        Returns:
            tuple[np.ndarray, int]: 音声データとサンプリングレート
        """
        sr = 24000
        payload = {
            "speakerUuid": speaker_uuid,
            "styleId": style_id,
            "text": text,
            "speedScale": 1.0,
            "volumeScale": 1.0,
            "pitchScale": 0.0,
            "intonationScale": 1.0,
            "prePhonemeLength": 0.1,
            "postPhonemeLength": 0.1,
            "outputSamplingRate": sr,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{self.port}/v1/synthesis", json=payload
            ) as response:
                if response.status != 200:
                    print(f"Error: {await response.text()}")
                    return None
                wav_data = await response.read()
        # WAVデータの読み込みは blocking な処理なので、to_thread で非同期に実行
        data, _sr = await asyncio.to_thread(self._read_wav, wav_data)
        assert _sr == sr
        return data, sr
