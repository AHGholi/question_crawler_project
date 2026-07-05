# question_generator\local_hf_qg.py

from __future__ import annotations

from dataclasses import dataclass

from .hf_pipe_factory import make_text2text_pipeline, Text2TextPipe
from .chunked_hf_qg_base import ChunkedHFQuestionGeneratorBase, CommonChunkedHFQGConfigProto


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

    per_chunk_min: int = 1
    per_chunk_max: int = 1
    max_chunks_used: int = 10

    second_pass_enabled: bool = True
    second_pass_limit: int = 8

    max_question_chars: int = 180
    min_question_chars: int = 18
    dedupe_jaccard: float = 0.85


class LocalHFQuestionGenerator(ChunkedHFQuestionGeneratorBase):
    backend_name = "local_hf"

    def __init__(self, config: LocalHFQGConfig | None = None):
        self._config = config or LocalHFQGConfig()
        self.pipe: Text2TextPipe = make_text2text_pipeline(
            model_name=self._config.model_name,
            device=self._config.device,
        )

    @property
    def config(self) -> CommonChunkedHFQGConfigProto:
        return self._config

    def _run_model(self, prompt: str) -> str:
        cfg = self._config
        out = self.pipe(
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            num_beams=cfg.num_beams,
            return_full_text=False,
        )
        if not out:
            return ""
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return str(out[0].get("generated_text", "")).strip()
        return str(out).strip()


def build_local_hf_qg(config: LocalHFQGConfig | None = None) -> LocalHFQuestionGenerator:
    return LocalHFQuestionGenerator(config=config)
