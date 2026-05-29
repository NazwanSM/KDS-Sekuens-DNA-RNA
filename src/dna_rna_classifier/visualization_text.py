from __future__ import annotations

def summarize_sequence_features(summary: dict) -> str:
    return (
        f"{summary.get('sequence_type', 'UNKNOWN')} sequence with length "
        f"{summary.get('length', 0)}, GC content {summary.get('gc_content', 0.0)}%, "
        f"and AT/AU content {summary.get('at_or_au_content', 0.0)}%."
    )

def summarize_mutation_analysis(result: dict) -> str:
    return str(result.get("summary", "No mutation summary available."))

def truncate_sequence(sequence: str, max_len: int = 120) -> str:
    if max_len <= 0:
        raise ValueError("max_len must be positive.")
    if len(sequence) <= max_len:
        return sequence
    if max_len <= 3:
        return sequence[:max_len]
    keep = (max_len - 3) // 2
    return f"{sequence[:keep]}...{sequence[-keep:]}"