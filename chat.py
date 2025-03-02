from langchain_ollama import ChatOllama
from voicevox import vv_synthesize
from coeiroink import ci_synthesize
from recog import SpeechRecognizer
import sounddevice as sd


def chat_start(cfg):
    llm = ChatOllama(**cfg.ollama)
    user_name = cfg.chat.user_name
    ai_name = cfg.chat.ai_name
    prompt = f"system: {cfg.chat.system_prompt}\n{cfg.chat.initial_message}"
    if cfg.chat.voice_input:
        recognizer = SpeechRecognizer(cfg.vosk.model_dir)

    print("Chat Start")
    while True:
        if cfg.chat.voice_input:
            print(f"{user_name}: ", end="", flush=True)
            user_input = recognizer.audio_input()
            print(user_input, flush=True)
        else:
            user_input = input(f"{user_name}: ")
        prompt += f"\n{user_name}: {user_input}\n{ai_name}: "
        response_stream = llm.stream(prompt)

        print(f"{ai_name}: ", end="", flush=True)
        sd.wait()
        answer = ""
        for chunk in response_stream:
            print(chunk.content, end="", flush=True)
            answer += chunk.content

            if answer and answer[-1] in cfg.chat.streaming_voice_output:
                sd.wait()
                if cfg.chat.voice_output == "voicevox":
                    data, sr = vv_synthesize(answer, **cfg.voicevox)
                    sd.play(data, sr)
                elif cfg.chat.voice_output == "coeiroink":
                    data, sr = ci_synthesize(answer, **cfg.coeiroink)
                    sd.play(data, sr)
                prompt += answer
                answer = ""
        print()
        if answer:
            sd.wait()
            if cfg.chat.voice_output == "voicevox":
                data, sr = vv_synthesize(answer, **cfg.voicevox)
                sd.play(data, sr)
            elif cfg.chat.voice_output == "coeiroink":
                data, sr = ci_synthesize(answer, **cfg.coeiroink)
                sd.play(data, sr)
            prompt += answer
