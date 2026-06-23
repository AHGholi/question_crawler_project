from typing import Any, List, Tuple, cast
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    """
    Converts tokenized text into TF-IDF vectors.
    Expects each document as List[str].
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=cast(Any, None),  # runtime-correct + type-checker-safe
            lowercase=False,
        )

    def fit_transform(self, documents: List[List[str]]):
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents: List[List[str]]):
        return self.vectorizer.transform(documents)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def extract_top_keywords(
        self,
        tfidf_matrix,
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        feature_names = self.get_feature_names()
        results: List[List[Tuple[str, float]]] = []

        for row in tfidf_matrix:
            scores = row.toarray().flatten()
            top_indices = scores.argsort()[::-1][:top_k]
            keywords = [
                (feature_names[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0
            ]
            results.append(keywords)

        return results
