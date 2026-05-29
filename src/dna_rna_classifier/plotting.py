from __future__ import annotations
import pandas as pd
from .features import nucleotide_counts, nucleotide_percentages, top_kmers

def composition_dataframe(sequence: str) -> pd.DataFrame:
    counts = nucleotide_counts(sequence)
    percentages = nucleotide_percentages(sequence)
    return pd.DataFrame(
        [
            {"base": base, "count": counts[base], "percentage": percentages[base]}
            for base in counts
        ]
    )

def kmer_dataframe(sequence: str, k: int = 3, top_n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(top_kmers(sequence, k=k, top_n=top_n), columns=["kmer", "count"])
