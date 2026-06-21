# question_generator\local_hf_qg.py

from __future__ import annotations

# pyright: reportGeneralTypeIssues=false

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, cast

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from question_generator.base import QGBackend, QGInput

logger = logging.getLogger(__name__)


@dataclass
class _LocalGenConfig:
    model_name: str = "google/flan-t5-large"
    device: int = -1
    max_new_tokens: int = 220
    temperature: float = 0.0
    do_sample: bool = False
    max_attempts: int = 4


class LocalHFQuestionGenerator(QGBackend):
    """
    Local Hugging Face question generator.

    Output contract (for adapter compatibility): List[str]
    - If answer exists, item is encoded as: "QUESTION || ANSWER"
    - Else item is just: "QUESTION"
    """

    _QA_SPLIT = " || "

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        *,
        device: int = -1,
        max_new_tokens: int = 220,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> None:
        self.cfg = _LocalGenConfig(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        self._mode: str = "causal"
        self._tokenizer: Any = None
        self._model: Any = None
        self._pipe: Any = None

        lowered = model_name.lower()
        likely_seq2seq = any(x in lowered for x in ("t5", "flan", "mt5", "bart", "pegasus"))

        if likely_seq2seq:
            self._mode = "seq2seq"
            self._tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name))
            self._model = cast(Any, AutoModelForSeq2SeqLM.from_pretrained(model_name))
            self._move_model_if_needed()
        else:
            self._mode = "causal"
            self._tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name))
            self._model = cast(Any, AutoModelForCausalLM.from_pretrained(model_name))
            self._move_model_if_needed()
            self._pipe = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device if device >= 0 else -1,
            )

        logger.info(
            "LocalHFQuestionGenerator initialized | model=%s | mode=%s | max_new_tokens=%s | temperature=%s | do_sample=%s",
            model_name,
            self._mode,
            max_new_tokens,
            temperature,
            do_sample,
        )

    def _move_model_if_needed(self) -> None:
        if self._model is None:
            return
        model = cast(Any, self._model)
        if self.cfg.device >= 0 and torch.cuda.is_available():
            self._model = model.to(f"cuda:{self.cfg.device}")
        else:
            self._model = model.to("cpu")

    def generate(self, data: QGInput) -> List[str]:
        n_questions = max(1, int(getattr(data, "num_questions", 10)))
        source_text = self._compose_source_text(data)
        if not source_text:
            return []

        collected: List[str] = []
        attempts = 0

        while len(collected) < n_questions and attempts < self.cfg.max_attempts:
            attempts += 1
            missing = n_questions - len(collected)

            if attempts == 1:
                prompt = self._build_prompt(source_text, n_questions=missing)
            else:
                prompt = self._build_retry_prompt(
                    source_text=source_text,
                    existing_items=collected,
                    missing=missing,
                )

            raw = self._generate_seq2seq(prompt) if self._mode == "seq2seq" else self._generate_causal(prompt)
            parsed = self._postprocess(raw, n_questions=missing, source_text=source_text)
            collected.extend(parsed)
            collected = self._dedupe_normalized(collected)

            logger.info(
                "QG attempt=%s requested=%s parsed=%s collected=%s raw_preview=%r",
                attempts,
                missing,
                len(parsed),
                len(collected),
                raw[:220],
            )

        return collected[:n_questions]

    def _generate_seq2seq(self, prompt: str) -> str:
        tokenizer = cast(Any, self._tokenizer)
        model = cast(Any, self._model)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": self.cfg.do_sample,
            "temperature": self.cfg.temperature if self.cfg.do_sample else None,
            "num_beams": 5,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.15,
            "length_penalty": 1.0,
            "early_stopping": True,
        }
        if gen_kwargs["temperature"] is None:
            gen_kwargs.pop("temperature")

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        return str(tokenizer.decode(output_ids[0], skip_special_tokens=True)).strip()

    def _generate_causal(self, prompt: str) -> str:
        pipe = cast(Any, self._pipe)
        out = pipe(
            prompt,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature if self.cfg.do_sample else None,
            return_full_text=False,
        )
        if not out:
            return ""
        return str(out[0].get("generated_text", "")).strip()

    @staticmethod
    def _compose_source_text(data: QGInput) -> str:
        topic = (getattr(data, "topic", None) or "").strip()
        summary = (getattr(data, "summary", None) or "").strip()
        sentences = list(getattr(data, "sentences", None) or [])
        keywords = list(getattr(data, "keywords", None) or [])

        clean_sents: List[str] = []
        for s in sentences:
            s = (s or "").strip()
            if len(s) < 35:
                continue
            if s.lower().startswith(("copyright", "all rights reserved", "cookie", "sign in")):
                continue
            clean_sents.append(s)

        clean_sents = clean_sents[:12]
        kw_part = ", ".join([k for k in keywords[:15] if isinstance(k, str) and len(k) > 2])

        parts: List[str] = []
        if topic:
            parts.append(f"Topic: {topic}")
        if summary:
            parts.append(f"Summary: {summary}")
        if kw_part:
            parts.append(f"Key terms: {kw_part}")
        if clean_sents:
            parts.append("Facts:\n" + "\n".join(f"- {x}" for x in clean_sents))

        return "\n\n".join(parts).strip()

    @staticmethod
    def _build_prompt(text: str, n_questions: int) -> str:
        return (
            "You are an exam question writer.\n"
            f"Write exactly {n_questions} QUESTION-ANSWER pairs based ONLY on the context.\n\n"
            "Format strictly:\n"
            "Q: <question text ending with ?>\n"
            "A: <short factual answer from context>\n\n"
            "Rules:\n"
            "- Use one Q/A pair per fact\n"
            "- No numbering\n"
            "- No generic wording\n"
            "- Questions must be specific and answerable from context\n\n"
            f"Context:\n{text}\n\n"
            "Output:\n"
        )

    @staticmethod
    def _build_retry_prompt(source_text: str, existing_items: Sequence[str], missing: int) -> str:
        existing_q = []
        for it in existing_items:
            q, _ = LocalHFQuestionGenerator._split_qa_item(it)
            if q:
                existing_q.append(q)
        existing = "\n".join(f"- {q}" for q in existing_q)

        return (
            "Generate additional QUESTION-ANSWER pairs from the SAME context.\n"
            f"Need exactly {missing} more pairs.\n"
            "Do not repeat existing questions.\n"
            "Format strictly:\n"
            "Q: ...?\n"
            "A: ...\n\n"
            f"Context:\n{source_text}\n\n"
            f"Existing questions:\n{existing}\n\n"
            "Additional output:\n"
        )

    @classmethod
    def _postprocess(cls, raw: str, n_questions: int, source_text: str) -> List[str]:
        pairs = cls._extract_qa_pairs(raw)

        # fallback: if model didn't follow Q/A format, parse as plain questions
        if not pairs:
            plain_q = cls._extract_plain_questions(raw)
            plain_q = [q for q in plain_q if cls._is_specific_question(q, source_text)]
            return cls._dedupe_normalized(plain_q)[:n_questions]

        items: List[str] = []
        for q, a in pairs:
            q = cls._normalize_question(q)
            a = cls._normalize_answer(a)
            if not cls._is_specific_question(q, source_text):
                continue
            if not a:
                # keep question even without answer (adapter may fill/leave empty)
                items.append(q)
            else:
                items.append(f"{q}{cls._QA_SPLIT}{a}")

        items = cls._dedupe_normalized(items)
        return items[:n_questions]

    @classmethod
    def _extract_qa_pairs(cls, raw: str) -> List[tuple[str, str]]:
        text = raw.strip()
        if not text:
            return []

        # Robust block parser for Q:/A:
        pattern = re.compile(
            r"(?:^|\n)\s*Q\s*[:\-]\s*(.+?)\s*\n\s*A\s*[:\-]\s*(.+?)(?=(?:\n\s*Q\s*[:\-])|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        out: List[tuple[str, str]] = []
        for m in pattern.finditer(text):
            q = " ".join(m.group(1).split())
            a = " ".join(m.group(2).split())
            out.append((q, a))
        return out

    @classmethod
    def _extract_plain_questions(cls, raw: str) -> List[str]:
        lines = [x.strip(" -\t\r\n") for x in raw.splitlines()]
        lines = [x for x in lines if x]

        out: List[str] = []
        for l in lines:
            l = re.sub(r"^\d+[\)\.\-]\s*", "", l).strip()
            if not l:
                continue
            if "?" in l:
                q = l[: l.find("?") + 1]
            else:
                q = f"{l}?"
            out.append(cls._normalize_question(q))
        return out

    @staticmethod
    def _normalize_question(q: str) -> str:
        q = " ".join(q.strip().split())
        q = re.sub(r"^[Qq]\s*[:\-]\s*", "", q).strip()
        if not q.endswith("?"):
            q = q.rstrip(".!") + "?"
        return q

    @staticmethod
    def _normalize_answer(a: str) -> str:
        a = " ".join(a.strip().split())
        a = re.sub(r"^[Aa]\s*[:\-]\s*", "", a).strip()
        return a

    @classmethod
    def _split_qa_item(cls, item: str) -> tuple[str, str]:
        if cls._QA_SPLIT in item:
            q, a = item.split(cls._QA_SPLIT, 1)
            return q.strip(), a.strip()
        return item.strip(), ""

    @staticmethod
    def _dedupe_normalized(items: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for x in items:
            norm = " ".join(x.lower().split())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(x.strip())
        return out

    @staticmethod
    def _is_specific_question(q: str, source_text: str) -> bool:
        ql = q.lower().strip()
        if len(ql) < 8:
            return False
        banned = ("key facts", "important aspects", "main idea")
        if any(b in ql for b in banned):
            return False
        return ql.endswith("?")
