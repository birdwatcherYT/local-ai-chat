from invoke import task
import asyncio


@task
def chat(c):
    """AIとのチャット"""
    from src.chat import chat_start

    # 非同期関数を実行
    asyncio.run(chat_start(c.config))


@task
def vv_list(c):
    """VOICEVOXの一覧を表示"""
    from src.tts.voicevox import VoiceVox

    VoiceVox().print_speakers()


@task
def ci_list(c):
    """COEIROINKの一覧を表示"""
    from src.tts.coeiroink import CoeiroInk

    CoeiroInk().print_speakers()


@task
def as_list(c):
    """AivisSpeechの一覧を表示"""
    from src.tts.aivisspeech import AivisSpeech

    AivisSpeech().print_speakers()
