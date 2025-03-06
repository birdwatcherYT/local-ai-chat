import asyncio
import sounddevice as sd
from langchain_ollama import ChatOllama
from invoke.config import Config
from .tts.base import TextToSpeech
from .asr.base import SpeechToText
from .tts.voicevox import VoiceVox
from .tts.coeiroink import CoeiroInk
from .tts.aivisspeech import AivisSpeech


async def playback_worker(queue: asyncio.Queue, asr: SpeechToText):
    """再生キューから順次オーディオデータを取り出して再生するワーカー"""
    while True:
        data, sr = await queue.get()
        if asr is not None:
            asr.pause()  # マイクをOFFにする
        sd.play(data, sr)
        await asyncio.to_thread(sd.wait)
        if asr is not None:
            asr.resume()  # 再生終了後にマイクをONにする
        queue.task_done()


async def synthesis_worker(
    synthesis_queue: asyncio.Queue,
    playback_queue: asyncio.Queue,
    engines: dict[str, TextToSpeech],
    ai_config: dict[str, Config],
):
    """合成キューから順次テキストを取り出して音声合成し、再生キューに投入するワーカー"""
    while True:
        name, text_segment = await synthesis_queue.get()
        cfg = ai_config[name]
        tts = engines[cfg["engine"]]
        data, sr = await tts.synthesize_async(text_segment, **cfg["config"])
        await playback_queue.put((data, sr))
        synthesis_queue.task_done()


def parse_message(message):
    parts = message.split(": ", 1)  # ": "で1回だけ分割
    if len(parts) == 2:
        name, text = parts
        return name, text
    else:
        return None, message


async def chat_start(cfg: Config):
    user_name = cfg.chat.user.name
    ai_names = {f"ai{i}_name": ai["name"] for i, ai in enumerate(cfg.chat.ai)}
    char_names = [user_name] + [ai["name"] for ai in cfg.chat.ai]

    # LLMの設定
    cfg.ollama.stop = [
        word.format(user_name=user_name, **ai_names) for word in cfg.ollama.stop
    ]
    llm = ChatOllama(**cfg.ollama)

    # 音声認識の設定
    asr: SpeechToText = None
    if cfg.chat.voice_input == "vosk":
        from .asr.vosk_asr import VoskASR

        asr = VoskASR(**cfg.vosk)
    elif cfg.chat.voice_input == "whisper":
        from .asr.whisper_asr import WhisperASR

        asr = WhisperASR(**cfg.whisper, **cfg.webrtcvad)

    # 音声合成の設定
    engines = {
        "voicevox": VoiceVox(),
        "coeiroink": CoeiroInk(),
        "aivisspeech": AivisSpeech(),
    }
    ai_config = {ai["name"]: ai["voice"] for ai in cfg.chat.ai}

    # 再生・合成用のグローバルなキューとワーカーを起動
    playback_queue = asyncio.Queue()
    synthesis_queue = asyncio.Queue()
    asyncio.create_task(playback_worker(playback_queue, asr))
    asyncio.create_task(
        synthesis_worker(synthesis_queue, playback_queue, engines, ai_config)
    )

    print(f"Chat Start: voice_input={cfg.chat.voice_input}")

    chara_prompt = "\n".join([f"{ai['name']}\n{ai['character']}" for ai in cfg.chat.ai])
    prompt = f"<system>\n{cfg.chat.system_prompt}\n{user_name}\n{cfg.chat.user.character}\n{chara_prompt}\n</system>\n{cfg.chat.initial_message}".format(
        user_name=user_name, **ai_names
    )
    turn = user_name
    print(f"{user_name}: ", end="", flush=True)
    # チャット全体をループで実行（各ターンごとにユーザー入力とテキスト生成を処理）
    while True:
        # ユーザー入力取得（音声入力の場合は asr.audio_input、テキストの場合は input()）
        if turn == user_name:
            if cfg.chat.voice_input:
                user_input = await asyncio.to_thread(asr.audio_input)
                print(user_input, flush=True)
            else:
                user_input = await asyncio.to_thread(input)

            prompt += f"{user_input}\n"
            turn = None

        # テキスト生成結果を受け取るためのキューを各ターンごとに作成
        text_queue = asyncio.Queue()

        async def process_text_queue():
            nonlocal prompt, turn
            answer = ""

            while True:
                chunk = await text_queue.get()
                if chunk is None:
                    break  # ストリーム終了の合図
                # 生成されたチャンクを即座に表示
                if turn:
                    print(chunk.content, end="", flush=True)
                answer += chunk.content
                # 指定された文字が現れたタイミングで音声合成
                if turn and answer and answer[-1] in cfg.chat.streaming_voice_output:
                    await synthesis_queue.put((turn, answer))
                    prompt += answer
                    answer = ""
                text_queue.task_done()
            if turn and answer:
                await synthesis_queue.put((turn, answer))
                prompt += answer
                answer = ""
            if turn:
                prompt += "\n"
                turn = None
                print()
            elif answer in char_names:
                turn = answer
                prompt += f"{turn}: "
                print(f"{turn}: ", end="", flush=True)
                # print(": ", end="", flush=True)
            # else:
            #     print(answer)
            #     print(prompt)
            #     exit(1)

        # テキスト処理タスクを開始
        processing_task = asyncio.create_task(process_text_queue())

        # 同期の llm.stream() を別スレッドで実行し、その結果を text_queue に投入する
        loop = asyncio.get_running_loop()

        def generate_text():
            for chunk in llm.stream(prompt):
                asyncio.run_coroutine_threadsafe(text_queue.put(chunk), loop)
            # ストリーム終了の合図として None を投入
            asyncio.run_coroutine_threadsafe(text_queue.put(None), loop)

        await asyncio.to_thread(generate_text)
        # テキスト処理タスクが完了するのを待つ
        await processing_task
