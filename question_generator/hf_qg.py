# question_generator\hf_qg.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

import requests

from .chunked_hf_qg_base import ChunkedHFQuestionGeneratorBase, CommonChunkedHFQGConfigProto

logger = logging.getLogger(__name__)


@dataclass
class HFQGConfig:
    api_token: str = ""
    model_name: str = "google/flan-t5-large"
    timeout: int = 60

    max_new_tokens: int = 96
    min_new_tokens: int = 12

    do_sample: bool = False
    temperature: float = 0.5
    top_p: float = 0.9
    num_beams: int = 4

    per_chunk_min: int = 1
    per_chunk_max: int = 1
    max_chunks_used: int = 10

    second_pass_enabled: bool = True
    second_pass_limit: int = 8

    max_question_chars: int = 180
    min_question_chars: int = 18
    dedupe_jaccard: float = 0.85


class HFQuestionGenerator(ChunkedHFQuestionGeneratorBase):
    backend_name = "hf_api"

    def __init__(
        self,
        api_token: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_beams: int | None = None,
        config: HFQGConfig | None = None,
    ):
        if config is None:
            config = HFQGConfig(
                api_token=api_token or "",
                model_name=model or "google/flan-t5-large",
                timeout=timeout or 60,
            )
            if max_new_tokens is not None:
                config.max_new_tokens = max_new_tokens
            if min_new_tokens is not None:
                config.min_new_tokens = min_new_tokens
            if do_sample is not None:
                config.do_sample = do_sample
            if temperature is not None:
                config.temperature = temperature
            if top_p is not None:
                config.top_p = top_p
            if num_beams is not None:
                config.num_beams = num_beams

        self._config = config
        self.endpoint = f"https://api-inference.huggingface.co/models/{self._config.model_name}"

    @property
    def config(self) -> CommonChunkedHFQGConfigProto:
        return self._config

    def _run_model(self, prompt: str) -> str:
        payload = self._call_hf(prompt)
        text = self._extract_text(payload)
        return text.strip()

    def _call_hf(self, prompt: str) -> Any:
        cfg = self._config
        headers = {
            "Authorization": f"Bearer {cfg.api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": cfg.max_new_tokens,
                "min_new_tokens": cfg.min_new_tokens,
                "do_sample": cfg.do_sample,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_beams": cfg.num_beams,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        logger.info("[QG][hf_api] POST model=%s prompt_chars=%d", cfg.model_name, len(prompt))

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=cfg.timeout,
        )

        logger.info(
            "[QG][hf_api] response status=%s content_type=%s",
            response.status_code,
            response.headers.get("content-type"),
        )

        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return response.text

    def _extract_text(self, payload: Any) -> str:
        if payload is None:
            logger.warning("[QG][hf_api] extract_text got None payload")
            return ""

        if isinstance(payload, str):
            return payload.strip()

        if isinstance(payload, list):
            parts: List[str] = []
            for item in payload:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    for key in ("generated_text", "summary_text", "text", "answer"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value.strip())
                            break
            text = "\n".join(parts).strip()
            if not text:
                logger.warning("[QG][hf_api] no text extracted from list payload")
            return text

        if isinstance(payload, dict):
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            for key in ("generated_text", "summary_text", "text", "answer"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            logger.warning("[QG][hf_api] no known text key in dict payload: %r", payload)
            return ""

        logger.warning("[QG][hf_api] unknown payload type: %s", type(payload).__name__)
        return ""
