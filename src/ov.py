from threading import Thread
from queue import Queue
from transformers import AutoTokenizer


# huggingface-cli login
# uv run optimum-cli export openvino --model google/gemma-2-2b-it --weight-format int4 --trust-remote-code gemma2


class OllamaResponse:
    """
    LangChain の ollama の仕様に合わせ、レスポンスは .content で生成文字列にアクセスできるようにするためのラッパークラスです。
    """

    def __init__(self, content: str):
        self.content = content


class OpenVinoLLM:
    """
    Gemma2Generator は OpenVINO 上で動作する Gemma-2 モデルによるテキスト生成を行います。
    streaming=True の場合は、生成されたトークンを逐次返します。
    """

    def __init__(
        self,
        model_path,
        tokenizer_model,
        device="GPU",
        stop_words=None,
    ):
        """
        初期化します。

        Args:
            model_path (str): エクスポート済みモデルのパス（例："gemma2"）。
            device (str): モデル実行デバイス ("CPU", "GPU", "NPU" など)。
            stop_words (list[str] or None): 生成を停止させる単語リスト。デフォルトは ["\n"]。
            tokenizer_model (str): トークナイザーとして利用する Hugging Face モデル。
        """
        if stop_words is None:
            stop_words = ["\n"]
        self.stop_words = stop_words
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        # 関数内でのインポートによりスレッドの多重生成を回避する
        import openvino_genai

        self.openvino_genai = openvino_genai
        self.pipe = self.openvino_genai.LLMPipeline(model_path, device)

    def _create_streamer(self):
        """
        内部的に StopWordsStreamer を生成します。
        このクラスは openvino_genai.StreamerBase を継承し、
        トークン毎にデコード＆キューに格納、停止条件をチェックします。
        """
        tokenizer = self.tokenizer
        stop_words = self.stop_words
        openvino_genai = self.openvino_genai

        class StopWordsStreamer(openvino_genai.StreamerBase):
            def __init__(self, stop_words=None):
                super().__init__()
                self.stop_words = stop_words if stop_words else []
                self.text_queue = Queue()
                self.stop_signal = object()
                self.generated_text = ""

            def __iter__(self):
                return self

            def __next__(self):
                value = self.text_queue.get()
                if value is self.stop_signal:
                    raise StopIteration()
                return value

            def put(self, token_id):
                # トークンIDをデコードして文字列に変換
                text = tokenizer.decode([token_id])
                self.generated_text += text
                # self.text_queue.put(text)
                # # 生成済みテキストに停止単語が含まれていれば終了
                # if any(
                #     stop_word in self.generated_text for stop_word in self.stop_words
                # ):
                #     self.end()
                # 生成済みテキストに停止単語が含まれていれば終了

                # 各ストップワードが新しいテキストに現れるかチェックし、最も早い出現位置を求める
                new_generated_text = self.generated_text + text
                cutoff_positions = []
                for stop_word in self.stop_words:
                    pos = new_generated_text.find(stop_word)
                    if pos != -1:
                        cutoff_positions.append(pos)
                if cutoff_positions:
                    cutoff = min(cutoff_positions)
                    # 既に生成済みの文字数を除いた、今回のトークン内で許容される部分を切り出す
                    allowed_new_text = new_generated_text[len(self.generated_text):cutoff]
                    self.generated_text = new_generated_text[:cutoff]
                    if allowed_new_text:
                        self.text_queue.put(allowed_new_text)
                    self.end()
                else:
                    self.generated_text = new_generated_text
                    self.text_queue.put(text)

            def end(self):
                self.text_queue.put(self.stop_signal)

        return StopWordsStreamer(stop_words=stop_words)

    def stream(self, prompt, max_new_tokens=200):
        """
        ストリーミングモードでテキスト生成を行い、生成されたトークンを逐次 yield します。

        Args:
            prompt (str): プロンプト文字列。
            max_new_tokens (int): 生成する新規トークン数の上限。

        Yields:
            OllamaResponse: 生成されたテキストの断片
        """
        streamer = self._create_streamer()
        generator_thread = Thread(
            target=self.pipe.generate,
            args=(prompt,),
            kwargs={"max_new_tokens": max_new_tokens, "streamer": streamer},
        )
        generator_thread.start()
        for token_text in streamer:
            yield OllamaResponse(token_text)
        generator_thread.join()

    def invoke(self, prompt, max_new_tokens=200) -> OllamaResponse:
        """
        テキスト生成を行います。

        Args:
            prompt (str): プロンプト文字列。
            max_new_tokens (int): 生成する新規トークン数の上限。

        Returns:
            OllamaResponse: 生成されたテキスト全体
        """
        return OllamaResponse(self.pipe.generate(prompt, max_new_tokens=max_new_tokens))
