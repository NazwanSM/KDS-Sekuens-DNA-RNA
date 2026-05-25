from __future__ import annotations
from .features import kmerize

class KmerTokenizer:
    """Callable k-mer tokenizer compatible with scikit-learn CountVectorizer."""

    def __init__(self, k: int = 6) -> None:
        """Store the k-mer size used by the tokenizer."""
        self.k = k

    def __call__(self, sequence: str) -> list[str]:
        """Tokenize a sequence into overlapping k-mers."""
        return kmerize(sequence, k=self.k)