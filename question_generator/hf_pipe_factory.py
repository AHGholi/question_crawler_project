"""Factory helpers for constructing local Hugging Face text-to-text pipelines.

The module wraps model loading so the question generator can use a simple
callable interface regardless of the underlying transformers implementation.
"""

from __future__ import annotations

from typing import Protocol

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Text2TextPipe(Protocol):
    """Callable protocol for text-to-text generation pipelines."""

    def __call__(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        min_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        temperature: float | None = None,
        top_p: float | None = None,
        return_full_text: bool = False,
    ) -> list[dict[str, str]]:
        ...


class _Seq2SeqRunner:
    """Small wrapper around a Hugging Face seq2seq model for question generation."""

    def __init__(self, model_name: str, device: int) -> None:
        """Load the tokenizer and model and move them to the requested device."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        min_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        temperature: float | None = None,
        top_p: float | None = None,
        return_full_text: bool = False,
    ) -> list[dict[str, str]]:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
        }

        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # keep same shape your local_hf_qg expects
        return [{"generated_text": text if not return_full_text else f"{prompt}{text}"}]


def make_text2text_pipeline(model_name: str, device: int) -> Text2TextPipe:
    """Create a callable text-to-text pipeline for the requested model."""
    return _Seq2SeqRunner(model_name=model_name, device=device)
