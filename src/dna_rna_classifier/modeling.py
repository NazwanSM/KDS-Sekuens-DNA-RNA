from __future__ import annotations
from .features import kmerize

class KmerTokenizer:
    def __init__(self, k: int = 6) -> None:
        self.k = k

    def __call__(self, sequence: str) -> list[str]:
        return kmerize(sequence, k=self.k)