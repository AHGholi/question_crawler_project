# processor\vectorizer.py
"""TF-IDF vectorization helpers for the processor subsystem.

This module turns tokenized text into sparse vectors that can be compared for
relevance and similarity.
"""

from typing import Any, List, Tuple, cast
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    """
    Converts tokenized text into TF-IDF vectors.
    Expects each document as List[str].
    """

    def __init__(self):
        """Create a TF-IDF vectorizer configured for token-list inputs."""
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=cast(Any, None),  # runtime-correct + type-checker-safe
            lowercase=False,
        )

    def fit_transform(self, documents: List[List[str]]):
        """Fit the vectorizer on documents and return their TF-IDF matrix."""
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents: List[List[str]]):
        """Transform new documents into the learned TF-IDF feature space."""
        return self.vectorizer.transform(documents)

    def get_feature_names(self) -> List[str]:
        """Return the learned vocabulary terms used by the vectorizer."""
        return self.vectorizer.get_feature_names_out().tolist()