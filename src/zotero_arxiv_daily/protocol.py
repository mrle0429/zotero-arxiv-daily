from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
from time import sleep
import re
import tiktoken
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from loguru import logger
import json
import random
RawPaperItem = TypeVar('RawPaperItem')

MAX_LLM_RETRIES = 3


def _chat_endpoint(openai_client: OpenAI) -> str:
    base_url = getattr(openai_client, "base_url", None)
    if base_url is None:
        return "chat/completions"
    return f"{str(base_url).rstrip('/')}/chat/completions"


def _retry_after_seconds(exc: RateLimitError) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


def _create_chat_completion_with_retries(openai_client: OpenAI, **kwargs):
    for attempt in range(MAX_LLM_RETRIES):
        try:
            return openai_client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            retry_after = _retry_after_seconds(exc)
            wait_time = (
                retry_after + random.uniform(0.2, 1.0)
                if retry_after is not None
                else 2 * (2 ** attempt) + random.uniform(0.2, 1.0)
            )
            logger.warning(
                f"Rate limited (attempt {attempt + 1}/{MAX_LLM_RETRIES}). "
                f"Retrying in {wait_time:.1f}s..."
            )
            if attempt == MAX_LLM_RETRIES - 1:
                raise
            sleep(wait_time)
        except (APITimeoutError, APIConnectionError) as exc:
            wait_time = 2 * (2 ** attempt) + random.uniform(0.2, 1.0)
            root_cause = getattr(exc, "__cause__", None)
            logger.warning(
                "Connection issue "
                f"(attempt {attempt + 1}/{MAX_LLM_RETRIES}, endpoint={_chat_endpoint(openai_client)}): {exc}. "
                f"Cause: {root_cause}. Retrying in {wait_time:.1f}s..."
            )
            if attempt == MAX_LLM_RETRIES - 1:
                raise
            sleep(wait_time)

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        response = _create_chat_completion_with_retries(
            openai_client,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr

    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = _create_chat_completion_with_retries(
                openai_client,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations

    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
