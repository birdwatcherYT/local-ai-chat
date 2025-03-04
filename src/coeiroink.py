import io
import aiohttp
import asyncio
import requests
import soundfile as sf
import numpy as np

# SEE http://localhost:50032/docs


def ci_print_speakers():
    """使えるキャラクター一覧を表示"""
    response = requests.get("http://localhost:50032/v1/speakers")

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


def ci_synthesize(
    text: str, speaker_uuid: str, style_id: int
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
    response = requests.post("http://localhost:50032/v1/synthesis", json=payload)
    if response.status_code == 200:
        # WAVデータをメモリから読み込む
        with io.BytesIO(response.content) as wav_file:
            data, _sr = sf.read(wav_file, dtype="float32")
        assert _sr == sr
        return data, sr
    else:
        print(f"Error: {response.text}")


async def ci_synthesize_async(
    text: str, speaker_uuid: str, style_id: int
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
            "http://localhost:50032/v1/synthesis", json=payload
        ) as response:
            if response.status != 200:
                print(f"Error: {await response.text()}")
                return None
            wav_data = await response.read()
    # WAVデータの読み込みは blocking な処理なので、to_thread で非同期に実行
    data, _sr = await asyncio.to_thread(_read_wav, wav_data)
    assert _sr == sr
    return data, sr


def _read_wav(wav_data: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(wav_data) as wav_file:
        data, sr = sf.read(wav_file, dtype="float32")
    return data, sr
