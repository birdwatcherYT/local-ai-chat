import io
import requests
import soundfile as sf

# SEE http://localhost:50032/docs


def print_speakers():
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


def ci_synthesize(text: str, speaker_uuid: str, style_id: int):
    """音声合成して再生する

    Args:
        text (str):
        speaker_uuid (str): _description_
        style_id (int): スタイルID

    Returns:
        _type_: _description_
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
        # sd.play(sig, sr)
        # sd.wait()
        # return sd
    else:
        print(f"Error: {response.text}")
