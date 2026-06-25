from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Sequence

from question_generator.base import QGBackend, QGInput
from .hf_pipe_factory import make_text2text_pipeline, Text2TextPipe

logger = logging.getLogger(__name__)


@dataclass
class LocalHFQGConfig:
    model_name: str = "google/flan-t5-large"
    device: int = -1

    max_new_tokens: int = 96
    min_new_tokens: int = 12

    do_sample: bool = False
    temperature: float = 0.5
    top_p: float = 0.9
    num_beams: int = 4

    # one-question-per-call strategy
    per_chunk_min: int = 1
    per_chunk_max: int = 1
    max_chunks_used: int = 10

    # two-pass fill if under target
    second_pass_enabled: bool = True
    second_pass_limit: int = 8

    max_question_chars: int = 180
    min_question_chars: int = 18
    dedupe_jaccard: float = 0.85


class LocalHFQuestionGenerator(QGBackend):
    def __init__(self, config: LocalHFQGConfig | None = None):
        self.config = config or LocalHFQGConfig()
        self.pipe: Text2TextPipe = make_text2text_pipeline(
            model_name=self.config.model_name,
            device=self.config.device,
        )

    # ✅ Protocol-compatible signature
    def generate(self, data: QGInput) -> List[str]:
        target = max(int(data.num_questions or 10), 1)

        chunks = self._prepare_chunks(data)
        if not chunks:
            logger.info("[QG][local_hf] no chunks and no fallback context")
            return []

        chunks = chunks[: self.config.max_chunks_used]
        budgets = self._allocate_budgets(target=target, num_chunks=len(chunks))
        logger.info(
            "[QG][local_hf] chunk_mode chunks=%d target=%d budgets=%s",
            len(chunks),
            target,
            budgets,
        )

        candidates: List[str] = []
        used_indices: List[int] = []

        # pass 1
        for i, (chunk, ask) in enumerate(zip(chunks, budgets), start=1):
            if ask <= 0:
                continue
            ask = max(1, ask)  # enforced one-question-per-call
            prompt = self._build_prompt(chunk=chunk, n=ask)
            raw = self._run_model(prompt)
            parsed = self._parse_output(raw)
            kept = self._postprocess(parsed)

            logger.info(
                "[QG][local_hf] pass1 chunk=%d/%d ask=%d parsed=%d kept=%d",
                i, len(chunks), ask, len(parsed), len(kept),
            )

            if kept:
                used_indices.append(i - 1)
                candidates.extend(kept)

        # pass 2 (top-up)
        deduped = self._dedupe(candidates)
        if self.config.second_pass_enabled and len(deduped) < target:
            remaining = target - len(deduped)
            pass2_indices = used_indices[: self.config.second_pass_limit]

            logger.info(
                "[QG][local_hf] pass2 start remaining=%d from_chunks=%s",
                remaining,
                pass2_indices,
            )

            for idx in pass2_indices:
                if len(deduped) >= target:
                    break

                chunk = chunks[idx]
                prompt = self._build_prompt(chunk=chunk, n=1)
                raw = self._run_model(prompt)
                parsed = self._parse_output(raw)
                kept = self._postprocess(parsed)

                logger.info(
                    "[QG][local_hf] pass2 chunk=%d/%d ask=1 parsed=%d kept=%d",
                    idx + 1, len(chunks), len(parsed), len(kept),
                )

                if kept:
                    candidates.extend(kept)
                    deduped = self._dedupe(candidates)

        final = self._dedupe(candidates)[:target]
        logger.info(
            "[QG][local_hf] summary candidates=%d deduped=%d final=%d target=%d",
            len(candidates), len(self._dedupe(candidates)), len(final), target
        )
        return final

    def _prepare_chunks(self, data: QGInput) -> List[str]:
        # Primary source: upstream chunks
        raw_chunks = list(data.chunks or [])
        chunks = [self._clean_chunk(c) for c in raw_chunks if c and c.strip()]
        chunks = [c for c in chunks if self._is_usable_chunk(c)]

        if chunks:
            return chunks

        # Fallback context from summary + sentences
        parts: List[str] = []
        if data.summary and data.summary.strip():
            parts.append(data.summary.strip())
        if data.sentences:
            parts.extend(s.strip() for s in data.sentences[:20] if s and s.strip())

        fallback = self._clean_chunk("\n".join(parts))
        if not fallback:
            return []

        # Try normal chunking on fallback text
        fb_chunks = self._chunk_context(fallback)
        if fb_chunks:
            return fb_chunks

        # Last resort: single chunk if informative enough
        if self._is_usable_chunk(fallback):
            return [fallback]

        return []

    def _clean_chunk(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _is_usable_chunk(self, text: str) -> bool:
        return len(text) >= 180 and text.count(" ") >= 25

    def _chunk_context(self, context: str) -> List[str]:
        # Split by paragraph-like breaks first
        parts = re.split(r"\n\s*\n+", context)
        out: List[str] = []

        for p in parts:
            t = self._clean_chunk(p)
            if not t:
                continue
            if self._is_usable_chunk(t):
                out.append(t)

        # fallback: sentence-window chunks if paragraph split is too poor
        if not out:
            txt = self._clean_chunk(context)
            sents = re.split(r"(?<=[.!?])\s+", txt)
            win = []
            cur = []
            cur_len = 0
            target_len = 500
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                cur.append(s)
                cur_len += len(s) + 1
                if cur_len >= target_len:
                    win.append(" ".join(cur).strip())
                    cur = []
                    cur_len = 0
            if cur:
                win.append(" ".join(cur).strip())

            out = [w for w in win if self._is_usable_chunk(w)]

        return out

    def _build_prompt(self, chunk: str, n: int) -> str:
        return (
            f"Generate up to {n} high-quality study question"
            f"{'' if n == 1 else 's'} from the passage.\n"
            "Rules:\n"
            "- Output ONLY questions.\n"
            "- One question per line.\n"
            "- Do NOT include answers.\n"
            "- Do NOT number questions.\n"
            "- Keep each question concise and complete.\n\n"
            f"Passage:\n{chunk}\n"
        )

    def _run_model(self, prompt: str) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=self.config.min_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_beams=self.config.num_beams,
            return_full_text=False,
        )
        if not out:
            return ""
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return str(out[0].get("generated_text", "")).strip()
        return str(out).strip()

    def _parse_output(self, text: str) -> List[str]:
        if not text:
            return []

        t = text.replace("\r", "\n")
        t = re.sub(r"(?<!\w)\s*(?:Q|Question)\s*\d+\s*[\).:\-]\s*", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"(?<=\?)\s+(?=(?:Q|Question)\s*\d+\s*[\).:\-])", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+\d+[\).:]\s+", "\n", t)
        t = re.sub(r"\s+-\s+", "\n", t)

        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        out: List[str] = []

        for ln in lines:
            ln = re.sub(r"^\s*(?:question\s*)?\d*[\).:\-]\s*", "", ln, flags=re.IGNORECASE)
            ln = re.sub(r"^\s*(?:q|question)\s*\d*\s*[:\-\).]\s*", "", ln, flags=re.IGNORECASE)
            ln = re.split(r"\b(?:answer|a)\s*:\s*", ln, flags=re.IGNORECASE)[0].strip()

            if "?" in ln:
                ln = ln.split("?", 1)[0].strip() + "?"

            out.append(ln.strip())

        return [x for x in out if x]

    def _postprocess(self, items: Sequence[str]) -> List[str]:
        out: List[str] = []
        for q in items:
            n = self._normalize_question(q)
            if not self._is_valid_question(n):
                continue
            out.append(n)
        return out

    def _normalize_question(self, q: str) -> str:
        q = re.sub(r"\s+", " ", q).strip()
        q = q.strip(" -•\t")
        q = re.sub(r"[.!]+$", "", q).strip()
        if q and not q.endswith("?"):
            q += "?"
        return q

    def _is_valid_question(self, q: str) -> bool:
        if not q:
            return False
        if "|" in q:
            return False
        if len(q) < self.config.min_question_chars or len(q) > self.config.max_question_chars:
            return False
        if q.count("?") != 1:
            return False
        if len(q.split()) < 6:
            return False
        # likely truncated endings
        if re.search(r"\b(of|and|or|to|with|for|in|on|by|from|the|a|an)\?$", q, flags=re.IGNORECASE):
            return False
        # citation-like garbage
        if re.search(r"“|”|\".*\b(using|studies|publication)\b", q, flags=re.IGNORECASE):
            return False

        bad = re.search(r"\b(?:no answer|not available|n/?a)\b", q, flags=re.IGNORECASE)
        if bad:
            return False
        return True

    def _dedupe(self, items: Sequence[str]) -> List[str]:
        out: List[str] = []
        for q in items:
            if not self._is_duplicate(q, out):
                out.append(q)
        return out

    def _is_duplicate(self, q: str, existing: Sequence[str]) -> bool:
        qt = self._token_set(q)
        if not qt:
            return True
        ql = q.lower().strip()
        for e in existing:
            el = e.lower().strip()
            if ql == el:
                return True
            et = self._token_set(e)
            inter = len(qt & et)
            union = len(qt | et) or 1
            j = inter / union
            if j >= self.config.dedupe_jaccard:
                return True
        return False

    def _token_set(self, s: str) -> set[str]:
        toks = re.findall(r"[a-zA-Z0-9]+", s.lower())
        return set(toks)

    def _allocate_budgets(self, target: int, num_chunks: int) -> List[int]:
        if num_chunks <= 0:
            return []

        budgets = [1 for _ in range(num_chunks)]

        if target <= 0:
            return budgets

        if target < num_chunks:
            active = max(1, target)
            budgets = [1 if i < active else 0 for i in range(num_chunks)]

        return budgets


def build_local_hf_qg(config: LocalHFQGConfig | None = None) -> LocalHFQuestionGenerator:
    return LocalHFQuestionGenerator(config=config)
