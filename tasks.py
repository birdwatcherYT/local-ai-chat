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
    from src.voicevox import vv_print_speakers
    vv_print_speakers()

@task
def ci_list(c):
    """COEIROINKの一覧を表示"""
    from src.coeiroink import ci_print_speakers
    ci_print_speakers()

@task
def as_list(c):
    """AivisSpeechの一覧を表示"""
    from src.aivisspeech import as_print_speakers
    as_print_speakers()
