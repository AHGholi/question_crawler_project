from __future__ import annotations

from typing import Any, List, cast

from transformers import pipeline as hf_pipeline
from transformers.pipelines.base import Pipeline

from question_generator.base import QGBackend, QGInput


class LocalHFQuestionGenerator(QGBackend):
    def __init__(
        self,
        model: str = "google/flan-t5-small",
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        do_sample: bool = False,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        pipeline_fn = cast(Any, hf_pipeline)

        self._pipe: Pipeline | None = None
        last_exc: Exception | None = None

        # Try tasks in order for max compatibility across transformers versions
        for task in ("text2text-generation", "translation", "text-generation"):
            try:
                self._pipe = cast(Pipeline, pipeline_fn(task, model=self.model))
                self._task = task
                break
            except Exception as exc:
                last_exc = exc

        if self._pipe is None:
            raise RuntimeError(
                f"Could not initialize HF pipeline for model={self.model}. "
                f"Tried tasks: text2text-generation, translation, text-generation. "
                f"Last error: {last_exc}"
            )

    def _build_prompt(self, data: QGInput) -> str:
        parts: List[str] = []
        if data.topic:
            parts.append(f"Topic: {data.topic}")
        if data.title:
            parts.append(f"Title: {data.title}")
        if data.summary:
            parts.append(f"Summary: {data.summary}")
        if data.keywords:
            parts.append("Keywords: " + ", ".join(data.keywords[:12]))
        if data.sentences:
            parts.append("Source sentences:\n" + "\n".join(f"- {s}" for s in data.sentences[:8]))

        n = max(1, int(data.num_questions))
        return (
            f"Generate {n} clear, non-duplicate study questions. Return one question per line.\n\n"
            + "\n".join(parts)
        )

    @staticmethod
    def _postprocess(text: str, n: int) -> List[str]:
        lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip()]
        out: List[str] = []
        seen = set()
        for ln in lines:
            if not ln.endswith("?"):
                ln = ln.rstrip(".") + "?"
            key = ln.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(ln)
            if len(out) >= n:
                break
        return out

    def _fallback(self, data: QGInput) -> List[str]:
        n = max(1, int(data.num_questions))
        seeds = (data.keywords[:n] if data.keywords else [data.topic or "the topic"])
        return [f"What is {s}?" for s in seeds[:n]]

    def generate(self, data: QGInput) -> List[str]:
        assert self._pipe is not None
        n = max(1, int(data.num_questions))
        prompt = self._build_prompt(data)

        result_any: Any = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
        )

        text = ""
        if isinstance(result_any, list) and result_any and isinstance(result_any[0], dict):
            first = result_any[0]
            text = str(
                first.get("generated_text")
                or first.get("translation_text")
                or first.get("text")
                or ""
            )

        qs = self._postprocess(text, n)
        return qs if qs else self._fallback(data)
