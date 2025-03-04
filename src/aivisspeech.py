import numpy as np

# vvとインターフェースが共通なのでそのまま利用する
from .voicevox import vv_print_speakers, vv_synthesize, vv_synthesize_async

# SEE http://localhost:10101/docs


def as_print_speakers():
    """使えるキャラクター一覧を表示"""
    vv_print_speakers(10101)


def as_synthesize(text: str, speaker_id: int) -> tuple[np.ndarray, int]:
    """音声合成して再生する

    Args:
        text (str): 音声合成したいテキスト
        speaker_id (int): キャラクターID

    Returns:
        tuple[np.ndarray, int]: 音声データとサンプリングレート
    """
    return vv_synthesize(text, speaker_id, 10101)


async def as_synthesize_async(text: str, speaker_id: int) -> tuple[np.ndarray, int]:
    """非同期で音声合成を行う

    Args:
        text (str): 音声合成したいテキスト
        speaker_id (int): キャラクターID

    Returns:
        tuple[np.ndarray, int]: 音声データとサンプリングレート
    """
    return await vv_synthesize_async(text, speaker_id, 10101)
