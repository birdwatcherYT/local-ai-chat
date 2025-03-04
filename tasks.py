from invoke import task
from invoke.config import Config
import asyncio


@task
def chat(c: Config):
    """AIとのチャット"""
    from src.chat import chat_start

    # 非同期関数を実行
    asyncio.run(chat_start(c.config))


@task
def vv_list(c: Config):
    """VOICEVOXの一覧を表示"""
    from src.tts.voicevox import VoiceVox

    VoiceVox().print_speakers()


@task
def vv_test(c: Config, text: str):
    """VOICEVOXで音声合成のテスト"""
    import sounddevice as sd
    from src.tts.voicevox import VoiceVox

    data, sr = VoiceVox().synthesize(text, **c.config.voicevox)
    sd.play(data, sr)
    sd.wait()


@task
def ci_list(c: Config):
    """COEIROINKの一覧を表示"""
    from src.tts.coeiroink import CoeiroInk

    CoeiroInk().print_speakers()


@task
def ci_test(c: Config, text: str):
    """COEIROINKで音声合成のテスト"""
    import sounddevice as sd
    from src.tts.coeiroink import CoeiroInk

    data, sr = CoeiroInk().synthesize(text, **c.config.coeiroink)
    sd.play(data, sr)
    sd.wait()


@task
def as_list(c: Config):
    """AivisSpeechの一覧を表示"""
    from src.tts.aivisspeech import AivisSpeech

    AivisSpeech().print_speakers()


@task
def as_test(c: Config, text: str):
    """AivisSpeechで音声合成のテスト"""
    import sounddevice as sd
    from src.tts.aivisspeech import AivisSpeech

    data, sr = AivisSpeech().synthesize(text, **c.config.aivisspeech)
    sd.play(data, sr)
    sd.wait()


@task
def whisper_test(c: Config, loop: bool = False):
    """Whisperのテスト"""
    from src.asr.whisper_asr import WhisperASR

    print("読み取り開始")
    asr = WhisperASR(**c.config.whisper, **c.config.webrtcvad)
    if loop:
        while True:
            text = asr.audio_input()
            print(text)
    else:
        text = asr.audio_input()
        print(text)


@task
def vosk_test(c: Config, loop: bool = False):
    """Voskのテスト"""
    from src.asr.vosk_asr import VoskASR

    print("読み取り開始")
    asr = VoskASR(**c.config.vosk)
    if loop:
        while True:
            text = asr.audio_input()
            print(text)
    else:
        text = asr.audio_input()
        print(text)
