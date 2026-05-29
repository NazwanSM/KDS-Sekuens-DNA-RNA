from __future__ import annotations
from collections import Counter
from .validation import detect_sequence_type, validate_sequence

def _bases_for_sequence(sequence: str) -> list[str]:
    sequence_type = detect_sequence_type(sequence)
    if sequence_type == "RNA":
        return ["A", "U", "G", "C", "N"]
    return ["A", "T", "G", "C", "N"]

def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

def nucleotide_counts(sequence: str) -> dict[str, int]:
    cleaned = validate_sequence(sequence)
    counts = Counter(cleaned)
    return {base: int(counts.get(base, 0)) for base in _bases_for_sequence(cleaned)}

def nucleotide_percentages(sequence: str) -> dict[str, float]:
    cleaned = validate_sequence(sequence)
    total = len(cleaned)
    counts = nucleotide_counts(cleaned)
    if total == 0:
        return {base: 0.0 for base in counts}
    return {base: round(count / total * 100, 2) for base, count in counts.items()}

def gc_content(sequence: str) -> float:
    cleaned = validate_sequence(sequence)
    if not cleaned:
        return 0.0
    gc_count = cleaned.count("G") + cleaned.count("C")
    return round(gc_count / len(cleaned) * 100, 2)

def at_or_au_content(sequence: str) -> float:
    cleaned = validate_sequence(sequence)
    if not cleaned:
        return 0.0
    if detect_sequence_type(cleaned) == "RNA":
        count = cleaned.count("A") + cleaned.count("U")
    else:
        count = cleaned.count("A") + cleaned.count("T")
    return round(count / len(cleaned) * 100, 2)

def kmerize(sequence: str, k: int = 3) -> list[str]:
    _validate_k(k)
    cleaned = validate_sequence(sequence)
    if len(cleaned) < k:
        return []
    return [cleaned[i : i + k] for i in range(len(cleaned) - k + 1)]

def kmer_counts(sequence: str, k: int = 3) -> dict[str, int]:
    return dict(Counter(kmerize(sequence, k=k)))

def kmer_frequencies(sequence: str, k: int = 3) -> dict[str, float]:
    counts = kmer_counts(sequence, k=k)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {kmer: round(count / total, 4) for kmer, count in counts.items()}

def top_kmers(sequence: str, k: int = 3, top_n: int = 10) -> list[tuple[str, int]]:
    if top_n <= 0:
        return []
    counts = kmer_counts(sequence, k=k)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]

def find_motifs(sequence: str, motifs: list[str]) -> list[dict]:
    cleaned = validate_sequence(sequence)
    hits: list[dict] = []

    for motif in motifs:
        cleaned_motif = validate_sequence(motif)
        start = 0
        while True:
            index = cleaned.find(cleaned_motif, start)
            if index == -1:
                break
            hits.append(
                {
                    "motif": cleaned_motif,
                    "start": index,
                    "end": index + len(cleaned_motif),
                    "match": cleaned[index : index + len(cleaned_motif)],
                }
            )
            start = index + 1

    return hits

def sequence_feature_summary(sequence: str, k: int = 3) -> dict:
    cleaned = validate_sequence(sequence)
    return {
        "sequence_type": detect_sequence_type(cleaned),
        "length": len(cleaned),
        "counts": nucleotide_counts(cleaned),
        "percentages": nucleotide_percentages(cleaned),
        "gc_content": gc_content(cleaned),
        "at_or_au_content": at_or_au_content(cleaned),
        "k": k,
        "unique_kmers": len(kmer_counts(cleaned, k=k)),
        "top_kmers": top_kmers(cleaned, k=k),
    }