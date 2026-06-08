from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class GeneratorConfig:
    """
    Shared, strongly-typed configuration for question generators.
    """
    strategy: str
    strategies: List[str]
    max_questions: int
    min_questions: int
    min_sentence_score: float
    max_keywords_per_question: int
    
    answer_max_chars: int
    confidence_decay: float

    # Neural QG configuration
    model_name: str
    input_format: str
    device: str
    max_input_length: int
    max_output_length: int
    num_beams: int
    num_return_sequences: int
    temperature: float
    top_p: float
    max_answers_per_sentence: int
    max_keyword_candidates: int = 20
    
