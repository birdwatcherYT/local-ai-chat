# local-ai-chat
すべてローカルで動作するAIチャット

## 環境構築
- `uv sync`
    - pythonと[uv](https://github.com/astral-sh/uv)インストール済みであることが前提
- ollamaで日本語対応モデルを使えるようにしておく
    - 例: `ollama run hf.co/mmnga/umiyuki-Umievo-itr012-Gleipnir-7B-gguf:Q8_0`
- 音声合成を使う場合（使う方をインストール）
    - [VOICEVOX](https://voicevox.hiroshiba.jp/)をインストール
    - [COEIROINK](https://coeiroink.com/download)をインストール
- 音声認識を使う場合
    - [VOSK Models](https://alphacephei.com/vosk/models)から`vosk-model-ja-0.22`をDLして展開
- `invoke.yaml`を環境に合わせる
    - LLMモデルの確認
    - 合成したいキャラクターのIDを確認
        - VOICEVOX: `uv run inv vv-list`（VOICEVOX GUI起動後）
        - COEIROINK: `uv run inv ci-list`（COEIROINK GUI起動後）
    - voskモデルへのパスを確認
- `uv run inv --list`
    - 動作確認
    - invoke.yamlの読み込みでエラーが出る場合は、invoke.yamlの文字コードをUTF-8にしてください(Windows用にSHIFT-JISにしています)

## 使い方
1. 音声合成を使う場合、裏でGUIを起動しておく
2. `uv run inv chat`
