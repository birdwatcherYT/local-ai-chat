import io
import aiohttp
import asyncio
import requests
import soundfile as sf
import numpy as np

# SEE http://localhost:50021/docs


def vv_print_speakers(port=50021):
    """使えるキャラクター一覧を表示"""
    response = requests.get(f"http://localhost:{port}/speakers")

    if response.status_code == 200:
        speakers = response.json()
        print(" id: name (style)")
        for speaker in speakers:
            for style in speaker["styles"]:
                print(f"{style['id']:3d}: {speaker['name']} ({style['name']})")
    else:
        print(f"Error: {response.status_code}")


def vv_synthesize(text: str, speaker_id: int, port=50021) -> tuple[np.ndarray, int]:
    """音声合成して再生する

    Args:
        text (str): 音声合成したいテキスト
        speaker_id (int): キャラクターID

    Returns:
        tuple[np.ndarray, int]: 音声データとサンプリングレート
    """
    # テキストから音声合成のためのクエリを作成
    query_payload = {"text": text, "speaker": speaker_id}
    query_response = requests.post(
        f"http://localhost:{port}/audio_query", params=query_payload
    )

    if query_response.status_code != 200:
        print(f"Error in audio_query: {query_response.text}")
        return

    query = query_response.json()
    sr = query["outputSamplingRate"]

    # クエリを元に音声データを生成
    synthesis_payload = {"speaker": speaker_id}
    synthesis_response = requests.post(
        f"http://localhost:{port}/synthesis", params=synthesis_payload, json=query
    )

    if synthesis_response.status_code == 200:
        # WAVデータをメモリから読み込む
        with io.BytesIO(synthesis_response.content) as wav_file:
            data, _sr = sf.read(wav_file, dtype="float32")
        assert _sr == sr
        return data, sr
    else:
        print(f"Error: {synthesis_response.text}")


async def vv_synthesize_async(
    text: str, speaker_id: int, port=50021
) -> tuple[np.ndarray, int]:
    """非同期で音声合成を行う

    Args:
        text (str): 音声合成したいテキスト
        speaker_id (int): キャラクターID

    Returns:
        tuple[np.ndarray, int]: 音声データとサンプリングレート
    """
    async with aiohttp.ClientSession() as session:
        query_payload = {"text": text, "speaker": speaker_id}
        async with session.post(
            f"http://localhost:{port}/audio_query", params=query_payload
        ) as resp:
            if resp.status != 200:
                print(f"Error in audio_query: {await resp.text()}")
                return None
            query = await resp.json()
            sr = query["outputSamplingRate"]

        synthesis_payload = {"speaker": speaker_id}
        async with session.post(
            f"http://localhost:{port}/synthesis", params=synthesis_payload, json=query
        ) as resp:
            if resp.status != 200:
                print(f"Error: {await resp.text()}")
                return None
            wav_data = await resp.read()
    # WAVデータの読み込みは blocking な処理なので、to_thread で非同期に実行
    data, _sr = await asyncio.to_thread(_read_wav, wav_data)
    assert _sr == sr
    return data, sr


def _read_wav(wav_data: bytes) -> tuple[np.ndarray, int]:
    with io.BytesIO(wav_data) as wav_file:
        data, sr = sf.read(wav_file, dtype="float32")
    return data, sr
