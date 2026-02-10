from llama_cpp import Llama
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from loguru import logger
from time import sleep
import random

GLOBAL_LLM = None

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,lang: str = "English"):
        if api_key:
            # Set reasonable timeout to prevent hanging requests
            self.llm = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    return response.choices[0].message.content
                except RateLimitError as e:
                    # Extract Retry-After header if available, otherwise exponential backoff
                    retry_after = getattr(e.response, 'headers', {}).get('retry-after') if e.response else None
                    if retry_after:
                        wait_time = float(retry_after) + random.uniform(1, 5)
                    else:
                        wait_time = min(30 * (2 ** attempt) + random.uniform(1, 5), 300)
                    logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.0f}s...")
                    if attempt == max_retries - 1:
                        raise
                    sleep(wait_time)
                except (APITimeoutError, APIConnectionError) as e:
                    wait_time = 5 * (2 ** attempt)
                    logger.warning(f"Connection issue (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    if attempt == max_retries - 1:
                        raise
                    sleep(wait_time)
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM