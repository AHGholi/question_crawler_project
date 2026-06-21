# question_generator/__init__.py

from .base import QGBackend, QGInput
from .adapter import PipelineQuestionGenerator
from .hf_qg import HFQuestionGenerator
from .local_hf_qg import LocalHFQuestionGenerator



__all__ = [
    "QGBackend",
    "QGInput",
    "PipelineQuestionGenerator",
    "HFQuestionGenerator",
    "LocalHFQuestionGenerator",
]
