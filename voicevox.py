import io
import requests
import soundfile as sf

# SEE http://localhost:50021/docs


def print_speakers():
    """使えるキャラクター一覧を表示"""
    response = requests.get("http://localhost:50021/speakers")

    if response.status_code == 200:
        speakers = response.json()
        print(" id: name (style)")
        for speaker in speakers:
            for style in speaker["styles"]:
                print(f"{style['id']:3d}: {speaker['name']} ({style['name']})")
    else:
        print(f"Error: {response.status_code}")


# 音声合成を行う関数
def vv_synthesize(text: str, speaker_id: int):
    """音声合成して再生する

    Args:
        text (str): 音声合成したいテキスト
        speaker_id (int): キャラクターID

    Returns:
        _type_: _description_
    """
    # テキストから音声合成のためのクエリを作成
    query_payload = {"text": text, "speaker": speaker_id}
    query_response = requests.post(
        "http://localhost:50021/audio_query", params=query_payload
    )

    if query_response.status_code != 200:
        print(f"Error in audio_query: {query_response.text}")
        return

    query = query_response.json()
    sr = query["outputSamplingRate"]

    # クエリを元に音声データを生成
    synthesis_payload = {"speaker": speaker_id}
    synthesis_response = requests.post(
        "http://localhost:50021/synthesis", params=synthesis_payload, json=query
    )

    if synthesis_response.status_code == 200:
        # WAVデータをメモリから読み込む
        with io.BytesIO(synthesis_response.content) as wav_file:
            data, _sr = sf.read(wav_file, dtype="float32")
        assert _sr == sr
        return data, sr
        # sd.play(data, sr)
        # sd.wait()
        # return sd
    else:
        print(f"Error: {synthesis_response.text}")
