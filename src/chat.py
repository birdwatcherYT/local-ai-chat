import asyncio
import sounddevice as sd
from langchain_ollama import ChatOllama
from .voicevox import vv_synthesize_async
from .coeiroink import ci_synthesize_async
from .aivisspeech import as_synthesize_async


async def playback_worker(queue: asyncio.Queue):
    """再生キューから順次オーディオデータを取り出して再生するワーカー"""
    while True:
        data, sr = await queue.get()
        sd.play(data, sr)
        await asyncio.to_thread(sd.wait)
        queue.task_done()


async def synthesis_worker(
    synthesis_queue: asyncio.Queue, cfg, playback_queue: asyncio.Queue
):
    """合成キューから順次テキストを取り出して音声合成し、再生キューに投入するワーカー"""
    while True:
        text_segment = await synthesis_queue.get()
        if cfg.chat.voice_output == "voicevox":
            data, sr = await vv_synthesize_async(text_segment, **cfg.voicevox)
        elif cfg.chat.voice_output == "coeiroink":
            data, sr = await ci_synthesize_async(text_segment, **cfg.coeiroink)
        elif cfg.chat.voice_output == "aivisspeech":
            data, sr = await as_synthesize_async(text_segment, **cfg.aivisspeech)
        await playback_queue.put((data, sr))
        synthesis_queue.task_done()


async def chat_start(cfg):
    llm = ChatOllama(**cfg.ollama)
    user_name = cfg.chat.user_name
    ai_name = cfg.chat.ai_name
    prompt = f"system: {cfg.chat.system_prompt}\n{cfg.chat.initial_message}"

    if cfg.chat.voice_input == "vosk":
        from .vosk_asr import VoskSpeechToText

        recognizer = VoskSpeechToText(cfg.vosk.model_dir)
    elif cfg.chat.voice_input == "whisper":
        from .whisper_asr import WhisperSpeechToText

        recognizer = WhisperSpeechToText(**cfg.whisper, **cfg.webrtcvad)

    # 再生・合成用のグローバルなキューとワーカーを起動
    playback_queue = asyncio.Queue()
    synthesis_queue = asyncio.Queue()
    asyncio.create_task(playback_worker(playback_queue))
    asyncio.create_task(synthesis_worker(synthesis_queue, cfg, playback_queue))

    print("Chat Start")

    # チャット全体をループで実行（各ターンごとにユーザー入力とテキスト生成を処理）
    while True:
        # ユーザー入力取得（音声入力の場合は recognizer.audio_input、テキストの場合は input()）
        if cfg.chat.voice_input:
            print(f"{user_name}: ", end="", flush=True)
            user_input = await asyncio.to_thread(recognizer.audio_input)
            print(user_input, flush=True)
        else:
            user_input = await asyncio.to_thread(input, f"{user_name}: ")

        # ユーザーの発話とそれに続く AI 応答の開始をプロンプトに追加
        prompt += f"\n{user_name}: {user_input}\n{ai_name}: "
        print(f"{ai_name}: ", end="", flush=True)

        # テキスト生成結果を受け取るためのキューを各ターンごとに作成
        text_queue = asyncio.Queue()

        async def process_text_queue():
            nonlocal prompt
            answer = ""
            while True:
                chunk = await text_queue.get()
                if chunk is None:
                    break  # ストリーム終了の合図
                # 生成されたチャンクを即座に表示
                print(chunk.content, end="", flush=True)
                answer += chunk.content
                # streaming_voice_output で指定された文字が現れたタイミングで音声合成要求
                if answer and answer[-1] in cfg.chat.streaming_voice_output:
                    await synthesis_queue.put(answer)
                    prompt += answer
                    answer = ""
                text_queue.task_done()
            if answer:
                await synthesis_queue.put(answer)
                prompt += answer
            print()

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
