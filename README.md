# local-ai-chat
すべてローカルで動作するAIチャット

## 環境構築
- `uv sync`
    - pythonと[uv](https://github.com/astral-sh/uv)インストール済みであることが前提
- ollamaで日本語対応モデルを使えるようにしておく
    - 例: `ollama run hf.co/mmnga/umiyuki-Umievo-itr012-Gleipnir-7B-gguf:Q8_0`
    - モデルが変わるとターン制御でうまくいかないことがあり、コードを修正する必要が出てきます
- 音声合成を使う場合（使うものを選択してインストール）
    - [VOICEVOX](https://voicevox.hiroshiba.jp/)をインストール
    - [COEIROINK](https://coeiroink.com/download)をインストール
    - [AivisSpeech](https://aivis-project.com/)をインストール
- 音声認識を使う場合
    - [VOSK Models](https://alphacephei.com/vosk/models)から`vosk-model-ja-0.22`をDLして展開
    - whisperを使う場合は設定不要（初回に自動ダウンロードされます）
- 動作確認
    - windows: `uv run inv -f invoke-shiftjis.yaml --list`
    - mac: `uv run inv -f invoke-utf8.yaml --list`

以下では`-f invoke-shiftjis.yaml`や`-f invoke-utf8.yaml`を省略して記述します。使う方を`invoke.yaml`としてリネームしてください。

- `invoke.yaml`を環境に合わせる
    - LLMモデルの確認
    - 合成したいキャラクターのIDを確認
        - VOICEVOX: `uv run inv vv-list`（VOICEVOX GUI起動後）
        - COEIROINK: `uv run inv ci-list`（COEIROINK GUI起動後）
        - AivisSpeech: `uv run inv as-list`（AivisSpeech GUI起動後）
    - voskモデルへのパスを確認

### Windows対応
- uv sync時のbuildエラー
    - visual studio build tools 2022でC++によるデスクトップ開発（MSVC、Windows11 SDK、CMake）をインストールしてからリトライ

### Mac対応
- uv sync時のvoskのエラー
    - pyproject.tomlからvoskを削除してからuv.sync


## 使い方
1. 音声合成を使う場合、裏でGUIを起動しておく
2. `uv run inv chat`
