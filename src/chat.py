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
        if cfg["engine"] is not None:
            tts = engines[cfg["engine"]]
            data, sr = await tts.synthesize_async(text_segment, **cfg["config"])
            await playback_queue.put((data, sr))
        synthesis_queue.task_done()


async def chat_start(cfg: Config):
    user_name = cfg.chat.user.name
    ai_names = {f"ai{i}_name": ai["name"] for i, ai in enumerate(cfg.chat.ai)}
    char_names = [ai["name"] for ai in cfg.chat.ai] + [user_name]

    # LLMの設定
    cfg.ollama.stop = [
        word.format(user_name=user_name, **ai_names) for word in cfg.ollama.stop
    ]
    llm = ChatOllama(**cfg.ollama)

    # 音声認識の設定
    asr: SpeechToText = None
    if cfg.chat.user.input == "vosk":
        from .asr.vosk_asr import VoskASR

        asr = VoskASR(**cfg.vosk)
    elif cfg.chat.user.input == "whisper":
        from .asr.whisper_asr import WhisperASR

        asr = WhisperASR(**cfg.whisper, **cfg.webrtcvad)

    # 音声合成の設定
    engines = {
        "voicevox": VoiceVox(),
        "coeiroink": CoeiroInk(),
        "aivisspeech": AivisSpeech(),
    }
    ai_config = {ai["name"]: ai["voice"] for ai in cfg.chat.ai}
    if cfg.chat.user.input == "ai":
        ai_config[user_name] = cfg.chat.user.voice

    # 再生・合成用のグローバルなキューとワーカーを起動
    playback_queue = asyncio.Queue()
    synthesis_queue = asyncio.Queue()
    asyncio.create_task(playback_worker(playback_queue, asr))
    asyncio.create_task(
        synthesis_worker(synthesis_queue, playback_queue, engines, ai_config)
    )

    print(f"Chat Start: user.input={cfg.chat.user.input}", flush=True)

    chara_prompt = "\n".join([f"{ai['name']}\n{ai['character']}" for ai in cfg.chat.ai])
    instruct_prompt = f"{cfg.chat.system_prompt}\n{user_name}\n{cfg.chat.user.character}\n{chara_prompt}".format(
        user_name=user_name, **ai_names
    )
    messages = cfg.chat.initial_message.format(user_name=user_name, **ai_names)
    cfg.chat.retry.turn = cfg.chat.retry.turn.format(user_name=user_name, **ai_names)
    cfg.chat.retry.prompt = cfg.chat.retry.prompt.format(
        user_name=user_name, **ai_names
    )

    prev_turn = None
    turn = cfg.chat.initial_turn.format(user_name=user_name, **ai_names)
    retry_num = 0

    print(f"{turn}: ", end="", flush=True)
    messages += f"{turn}: "
    summary = ""
    prompt = ""
    # チャット全体をループで実行（各ターンごとにユーザー入力とテキスト生成を処理）
    while True:
        # ユーザー入力取得（音声入力の場合は asr.audio_input、テキストの場合は input()）
        if turn == user_name and cfg.chat.user.input != "ai":
            if cfg.chat.user.input == "text":
                user_input = await asyncio.to_thread(input)
            else:
                user_input = await asyncio.to_thread(asr.audio_input)
                print(user_input, flush=True)

            messages += f"{user_input}\n"
            if len(char_names) == 2:
                prev_turn = turn
                turn = char_names[0]
                messages += f"{turn}: "
                print(f"{turn}: ", end="", flush=True)
            else:
                prev_turn = turn
                turn = None

        # テキスト生成結果を受け取るためのキューを各ターンごとに作成
        text_queue = asyncio.Queue()

        async def process_text_queue():
            nonlocal messages, turn, prev_turn, retry_num
            answer = ""

            while True:
                chunk = await text_queue.get()
                if chunk is None:
                    break  # ストリーム終了の合図
                # 生成されたチャンクを即座に表示
                if turn:
                    print(chunk.content, end="", flush=True)
                elif cfg.chat.debug:
                    print("debug: ", chunk.content, flush=True)
                answer += chunk.content
                # 指定された文字が現れたタイミングで音声合成
                if turn and answer and answer[-1] in cfg.chat.streaming_voice_output:
                    await synthesis_queue.put((turn, answer))
                    messages += answer
                    answer = ""
                text_queue.task_done()
            if turn and answer:
                await synthesis_queue.put((turn, answer))
                messages += answer
                answer = ""

            if turn:
                # 上記でメッセージ追記後の改行
                messages += "\n"
                print()
                if len(char_names) == 2:
                    prev_turn = turn
                    turn = char_names[1]
                    messages += f"{turn}: "
                    print(f"{turn}: ", end="", flush=True)
                else:
                    prev_turn = turn
                    turn = None
                retry_num = 0
            elif answer in char_names and prev_turn != answer:
                turn = answer
                messages += f"{turn}: "
                print(f"{turn}: ", end="", flush=True)
                retry_num = 0
            elif retry_num >= cfg.chat.retry.num:
                # 場面切り替わりやナレーションが入る場合の対策
                if cfg.chat.retry.prompt:
                    messages += f"[INST]{cfg.chat.retry.prompt}[/INST]\n"
                    if cfg.chat.debug:
                        print(f"debug: 指示追加", flush=True)
                if cfg.chat.retry.turn:
                    prev_turn = None
                    turn = cfg.chat.retry.turn
                    messages += f"{turn}: "
                    print(f"{turn}: ", end="", flush=True)
                    if cfg.chat.debug:
                        print(f"debug: 強制ターン変更", flush=True)
                retry_num = 0
            else:
                retry_num += 1
                if cfg.chat.debug:
                    print(f"debug: [{retry_num}]", answer, flush=True)

        # テキスト処理タスクを開始
        processing_task = asyncio.create_task(process_text_queue())

        # 同期の llm.stream() を別スレッドで実行し、その結果を text_queue に投入する
        loop = asyncio.get_running_loop()

        prompt = f"[INST]\n{instruct_prompt}\n{summary}\n[/INST]\n{messages}"

        def generate_text():
            for chunk in llm.stream(prompt):
                asyncio.run_coroutine_threadsafe(text_queue.put(chunk), loop)
            # ストリーム終了の合図として None を投入
            asyncio.run_coroutine_threadsafe(text_queue.put(None), loop)

        await asyncio.to_thread(generate_text)
        # テキスト処理タスクが完了するのを待つ
        await processing_task

        def run_summary():
            nonlocal messages, summary, prompt
            print("要約中", flush=True)
            prompt = f"[INST]\n{instruct_prompt}\n{summary}\n[/INST]\n{messages}"
            # resp = llm.invoke(f"{messages}\n上記会話を1行で要約してください。\n")
            resp = llm.invoke(
                f"{prompt}\n[INST]上記会話を1行で要約してください。[/INST]\n"
            )
            summary = f"これまでの要約: {resp.content}"
            print(summary, flush=True)
            if cfg.chat.debug:
                print("debug:", summary, flush=True)
            message_list = messages.split("\n")
            cut_message_list = message_list[-cfg.chat.summary.tail :]
            if cfg.chat.debug:
                print(
                    "debug:", len(message_list), "->", len(cut_message_list), flush=True
                )
            messages = "\n".join(cut_message_list)
            prompt = f"[INST]\n{instruct_prompt}\n{summary}\n[/INST]\n{messages}"

        if len(messages) > cfg.chat.summary.length:
            await asyncio.to_thread(run_summary)
