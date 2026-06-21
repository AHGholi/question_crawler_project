# question_generator/hf_qg.py

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional

import requests

from .base import QGBackend, QGInput


class HFQuestionGenerator(QGBackend):
    """
    Hugging Face Inference API based question generator.

    Works well with instruct/text-generation models.
    Example model:
      mistralai/Mistral-7B-Instruct-v0.2
      google/flan-t5-large
    """

    def __init__(
        self,
        *,
        api_token: str,
        model: str = "google/flan-t5-large",
        timeout: int = 60,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
    ) -> None:
        if not api_token:
            raise ValueError("HF API token is required.")
        self.api_token = api_token
        self.model = model
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.url = f"https://api-inference.huggingface.co/models/{self.model}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _build_prompt(self, data: QGInput) -> str:
        # keep prompt concise and deterministic
        keywords = ", ".join(data.keywords[:20]) if data.keywords else ""
        sentences = "\n".join(f"- {s}" for s in data.sentences[:20])

        return (
            f"Generate exactly {data.num_questions} clear, non-duplicate questions.\n"
            f"Return only numbered questions, no explanations.\n\n"
            f"Topic: {data.topic}\n"
            f"Title: {data.title or ''}\n"
            f"Summary: {data.summary or ''}\n"
            f"Keywords: {keywords}\n"
            f"Source Sentences:\n{sentences}\n"
        )

    def _call_hf(self, prompt: str) -> Any:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        resp = requests.post(
            self.url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_text(response_json: Any) -> str:
        # HF responses vary by model/task
        if isinstance(response_json, list) and response_json:
            item = response_json[0]
            if isinstance(item, dict):
                if "generated_text" in item:
                    return str(item["generated_text"])
                if "summary_text" in item:
                    return str(item["summary_text"])
        if isinstance(response_json, dict):
            if "generated_text" in response_json:
                return str(response_json["generated_text"])
            if "error" in response_json:
                raise RuntimeError(f"HF inference error: {response_json['error']}")
        return str(response_json)

    @staticmethod
    def _parse_questions(text: str, expected: int) -> List[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned: List[str] = []

        # accept numbered or bullet lines
        for ln in lines:
            q = re.sub(r"^\s*(?:\d+[\).\-\:]\s*|[-*]\s*)", "", ln).strip()
            if q.endswith("?") and len(q) > 6:
                cleaned.append(q)
            elif len(q.split()) >= 4:
                # force as question if model forgot '?'
                cleaned.append(q.rstrip(".") + "?")

        # de-duplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for q in cleaned:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(q)

        return uniq[:expected]

    def generate(self, data: QGInput) -> List[str]:
        prompt = self._build_prompt(data)
        raw = self._call_hf(prompt)
        text = self._extract_text(raw)
        questions = self._parse_questions(text, expected=data.num_questions)

        if not questions:
            # fallback deterministic output
            questions = [
                f"What is the main idea of {data.topic}?"
            ]
            for kw in data.keywords[: max(0, data.num_questions - 1)]:
                questions.append(f"How does '{kw}' relate to {data.topic}?")

        return questions[: data.num_questions]
