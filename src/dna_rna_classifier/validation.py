from __future__ import annotations
import re

DNA_SYMBOLS = set("ATGCN")
RNA_SYMBOLS = set("AUGCN")
WHITESPACE_RE = re.compile(r"\s+")

def clean_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("Sequence must be a string.")
    return WHITESPACE_RE.sub("", sequence).upper()

def detect_sequence_type(sequence: str) -> str:
    cleaned = clean_sequence(sequence)
    if not cleaned:
        return "INVALID"

    symbols = set(cleaned)
    allowed = DNA_SYMBOLS | RNA_SYMBOLS
    if not symbols.issubset(allowed):
        return "INVALID"
    if "T" in symbols and "U" in symbols:
        return "INVALID"
    if symbols.issubset(DNA_SYMBOLS) and "T" in symbols:
        return "DNA"
    if symbols.issubset(RNA_SYMBOLS) and "U" in symbols:
        return "RNA"
    if symbols.issubset(DNA_SYMBOLS & RNA_SYMBOLS):
        return "AMBIGUOUS"
    return "INVALID"

def validate_sequence(sequence: str, allow_ambiguous: bool = True) -> str:
    cleaned = clean_sequence(sequence)
    sequence_type = detect_sequence_type(cleaned)
    if sequence_type == "INVALID":
        raise ValueError(
            "Invalid sequence. Use DNA bases A/T/G/C/N or RNA bases A/U/G/C/N, "
            "without mixing T and U."
        )
    if sequence_type == "AMBIGUOUS" and not allow_ambiguous:
        raise ValueError("Sequence is ambiguous because it contains no T or U.")
    return cleaned

def parse_fasta_text(text: str) -> list[tuple[str, str]]:
    if not isinstance(text, str):
        raise TypeError("FASTA input must be a string.")

    stripped = text.strip()
    if not stripped:
        return []

    if not stripped.startswith(">"):
        return [("sequence_1", validate_sequence(stripped))]

    records: list[tuple[str, str]] = []
    current_id: str | None = None
    current_lines: list[str] = []

    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, validate_sequence("".join(current_lines))))
            current_id = line[1:].strip().split()[0] or f"sequence_{len(records) + 1}"
            current_lines = []
        else:
            current_lines.append(line)

    if current_id is not None:
        records.append((current_id, validate_sequence("".join(current_lines))))
    return records

