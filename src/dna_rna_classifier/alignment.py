from __future__ import annotations

import numpy as np
from .validation import validate_sequence

def needleman_wunsch_matrix(
    seq1: str,
    seq2: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
) -> np.ndarray:
    clean1 = validate_sequence(seq1)
    clean2 = validate_sequence(seq2)
    matrix = np.zeros((len(clean1) + 1, len(clean2) + 1), dtype=int)

    for i in range(1, len(clean1) + 1):
        matrix[i, 0] = matrix[i - 1, 0] + gap
    for j in range(1, len(clean2) + 1):
        matrix[0, j] = matrix[0, j - 1] + gap

    for i, base1 in enumerate(clean1, start=1):
        for j, base2 in enumerate(clean2, start=1):
            diagonal = matrix[i - 1, j - 1] + (match if base1 == base2 else mismatch)
            up = matrix[i - 1, j] + gap
            left = matrix[i, j - 1] + gap
            matrix[i, j] = max(diagonal, up, left)

    return matrix

def needleman_wunsch(
    seq1: str,
    seq2: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
) -> dict:
    clean1 = validate_sequence(seq1)
    clean2 = validate_sequence(seq2)
    matrix = needleman_wunsch_matrix(clean1, clean2, match=match, mismatch=mismatch, gap=gap)

    i = len(clean1)
    j = len(clean2)
    aligned1: list[str] = []
    aligned2: list[str] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match if clean1[i - 1] == clean2[j - 1] else mismatch
            if matrix[i, j] == matrix[i - 1, j - 1] + score:
                aligned1.append(clean1[i - 1])
                aligned2.append(clean2[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and matrix[i, j] == matrix[i - 1, j] + gap:
            aligned1.append(clean1[i - 1])
            aligned2.append("-")
            i -= 1
        elif j > 0:
            aligned1.append("-")
            aligned2.append(clean2[j - 1])
            j -= 1
        else:
            aligned1.append(clean1[i - 1])
            aligned2.append("-")
            i -= 1

    aligned_seq1 = "".join(reversed(aligned1))
    aligned_seq2 = "".join(reversed(aligned2))
    return {
        "aligned_seq1": aligned_seq1,
        "aligned_seq2": aligned_seq2,
        "score": int(matrix[-1, -1]),
        "score_matrix_shape": tuple(matrix.shape),
        "metrics": alignment_metrics(aligned_seq1, aligned_seq2),
    }

def alignment_metrics(aligned_seq1: str, aligned_seq2: str) -> dict:
    if len(aligned_seq1) != len(aligned_seq2):
        raise ValueError("Aligned sequences must have the same length.")
    if not aligned_seq1:
        return {"matches": 0, "mismatches": 0, "gaps": 0, "identity_percentage": 0.0}

    matches = 0
    mismatches = 0
    gaps = 0
    for base1, base2 in zip(aligned_seq1, aligned_seq2):
        if base1 == "-" or base2 == "-":
            gaps += 1
        elif base1 == base2:
            matches += 1
        else:
            mismatches += 1

    identity = round(matches / len(aligned_seq1) * 100, 2)
    return {
        "matches": matches,
        "mismatches": mismatches,
        "gaps": gaps,
        "identity_percentage": identity,
    }

def format_alignment(aligned_seq1: str, aligned_seq2: str, line_length: int = 60) -> str:
    if len(aligned_seq1) != len(aligned_seq2):
        raise ValueError("Aligned sequences must have the same length.")
    if line_length <= 0:
        raise ValueError("line_length must be positive.")

    blocks: list[str] = []
    for start in range(0, len(aligned_seq1), line_length):
        part1 = aligned_seq1[start : start + line_length]
        part2 = aligned_seq2[start : start + line_length]
        markers = "".join(
            "|" if a == b and a != "-" else " " if "-" in {a, b} else "."
            for a, b in zip(part1, part2)
        )
        blocks.append(f"REF  {part1}\n     {markers}\nMUT  {part2}")
    return "\n\n".join(blocks)