"""
processor/text_processing.py

Provides text preprocessing utilities:
- normalization (lowercasing, punctuation removal)
- tokenization
- stopword removal
- lemmatization
"""

import re
import string
import spacy
from typing import List
# Load small English model (do once globally)
# This model includes tokenizer, tagger, and lemmatizer
_nlp = spacy.load("en_core_web_sm")

class TextProcessor:
    """
    A class for preprocessing text for further NLP analysis.
    """


    def __init__(self, nlp=None):
        self.nlp = nlp or _nlp
        

    def preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize, lemmatize, and remove stopwords and punctuation.
        """
        doc = self.nlp(text.lower())

        return [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.is_alpha
        ]

    def normalize_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove URLs, digits, and excessive spaces.
        """
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)    # remove URLs
        text = re.sub(r"\d+", "", text)               # remove digits
        text = re.sub(r'\s+', ' ', text).strip()      # collapse multiple spaces
        return text

    def clean_punctuation(self, text: str) -> str:
        """
        Remove punctuation safely from text.
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str):
        """
        Tokenize text using spaCy.
        Returns a list of token objects.
        """
        doc = _nlp(text)
        return [token for token in doc]

    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens and remove stopwords/non-alphabetic terms.
        Returns a list of lemmatized words.
        """
        processed = [
            token.lemma_ for token in tokens
            if not token.is_stop and token.is_alpha
        ]
        return processed

    def process(self, text: str):
        """
        Main pipeline call: full text preprocessing
        -> normalized text -> tokens -> lemmas
        Returns list of processed tokens.
        """
        normalized = self.normalize_text(text)
        cleaned = self.clean_punctuation(normalized)
        tokens = self.tokenize(cleaned)
        lemmas = self.lemmatize_tokens(tokens)
        return lemmas

_default_processor = TextProcessor(_nlp)

def preprocess_text(text: str) -> List[str]:
    """
    Functional helper so callers/tests can preprocess without instantiating the class.
    """
    return _default_processor.preprocess_text(text)

if __name__ == "__main__":
    sample = "Neural networks are systems inspired by the human brain. They learn from data!"
    processor = TextProcessor()
    result = processor.process(sample)
    print(f"Processed tokens: {result}")
