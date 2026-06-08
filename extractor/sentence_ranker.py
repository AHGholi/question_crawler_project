# extractor/sentence_ranker.py

from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


from processor.text_processing import TextProcessor


class SentenceRanker:
    """
    Split documents into sentences and rank them by relevance to a query.
    """

    def __init__(self):
        self.text_processor = TextProcessor()

        # Use tokenizer from TextProcessor for consistency
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.text_processor.preprocess_text,
            lowercase=False
        )

    def _split_sentences(self, documents: List[str]) -> List[str]:
        """
        Split documents into sentences using spaCy.
        """
        sentences: List[str] = []

        for doc in self.text_processor.nlp.pipe(documents):
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(sent_text) > 5:
                    sentences.append(sent_text)

        return sentences

    def rank_sentences(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Rank sentences extracted from documents by relevance to the query.
        """
        if not documents:
            return []

        sentences = self._split_sentences(documents)

        if not sentences:
            return []

        corpus = [query] + sentences

        tfidf = self.vectorizer.fit_transform(corpus)

        query_vec = csr_matrix(tfidf.getrow(0))
        rows = [tfidf.getrow(i) for i in range(1, tfidf.shape[0])]
        sentence_vecs = csr_matrix(vstack(rows))

        scores = cosine_similarity(query_vec, sentence_vecs)[0]

        ranked = sorted(
            zip(sentences, scores),
            key=lambda x: x[1],
            reverse=True
        )

        if min_score > 0.0:
            ranked = [(s, sc) for s, sc in ranked if sc >= min_score]

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked
