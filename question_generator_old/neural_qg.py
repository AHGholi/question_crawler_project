#question_generator\neural_qg.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, cast

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass(slots=True)
class NeuralQGEngine:
    model_name: str
    input_format: str = "prepend"  # "prepend" or "highlight"
    device: str = "auto"           # "auto", "cuda", "cpu"
    max_input_length: int = 256
    max_output_length: int = 64
    num_beams: int = 2
    num_return_sequences: int = 1
    temperature: float = 1.0
    top_p: float = 1.0

    _tokenizer: PreTrainedTokenizerBase | None = None
    _model: PreTrainedModel | None = None
    _resolved_device: torch.device | None = None

    def _resolve_device(self) -> torch.device:
        if self.device and self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        self._resolved_device = self._resolve_device()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self._tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        self._model = cast(PreTrainedModel, model)

        self._model.to(self._resolved_device)
        self._model.eval()

    def build_input(self, answer: str, context: str) -> str:
        answer = (answer or "").strip()
        context = (context or "").strip()

        if self.input_format == "highlight":
            lowered = context.lower()
            ans_lower = answer.lower()
            idx = lowered.find(ans_lower)
            if idx >= 0:
                before = context[:idx]
                matched = context[idx : idx + len(answer)]
                after = context[idx + len(answer) :]
                return f"{before}<hl> {matched} <hl>{after}"
            return f"<hl> {answer} <hl> {context}"

        return f"answer: {answer}  context: {context}"

    def generate(self, inputs: List[str]) -> List[str]:
        if not inputs:
            return []

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._resolved_device is not None

        enc = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )

        enc = cast(BatchEncoding, enc)
        input_ids = enc["input_ids"].to(self._resolved_device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._resolved_device)

        gen_kwargs: dict[str, Any] = {
            "max_length": self.max_output_length,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
        }

        if self.temperature and (self.temperature != 1.0 or self.top_p < 1.0):
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [text.strip() for text in decoded]
