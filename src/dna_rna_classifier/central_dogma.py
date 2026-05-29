from __future__ import annotations
from .validation import detect_sequence_type, validate_sequence

DNA_BASES = {"A", "T", "G", "C", "N"}
RNA_BASES = {"A", "U", "G", "C", "N"}
DNA_COMPLEMENT = str.maketrans({"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"})
RNA_COMPLEMENT = str.maketrans({"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"})
STOP_CODONS_RNA = {"UAA", "UAG", "UGA"}

STANDARD_RNA_CODON_TABLE = {
    "UUU": "F",
    "UUC": "F",
    "UUA": "L",
    "UUG": "L",
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",
    "UAU": "Y",
    "UAC": "Y",
    "UAA": "*",
    "UAG": "*",
    "UGU": "C",
    "UGC": "C",
    "UGA": "*",
    "UGG": "W",
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAU": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

def dna_to_rna(dna: str) -> str:
    cleaned = validate_sequence(dna)
    if detect_sequence_type(cleaned) == "RNA":
        raise ValueError("Expected DNA sequence, received RNA sequence.")
    return cleaned.replace("T", "U")

def rna_to_dna(rna: str) -> str:
    cleaned = validate_sequence(rna)
    if detect_sequence_type(cleaned) == "DNA":
        raise ValueError("Expected RNA sequence, received DNA sequence.")
    return cleaned.replace("U", "T")

def get_complement(sequence: str) -> str:
    cleaned = validate_sequence(sequence)
    sequence_type = detect_sequence_type(cleaned)
    table = RNA_COMPLEMENT if sequence_type == "RNA" else DNA_COMPLEMENT
    return cleaned.translate(table)

def get_reverse_complement(sequence: str) -> str:
    return get_complement(sequence)[::-1]

def transcribe(dna: str) -> str:
    return dna_to_rna(dna)

def split_codons(sequence: str, frame: int = 0) -> list[str]:
    if frame not in {0, 1, 2}:
        raise ValueError("Reading frame must be 0, 1, or 2.")
    cleaned = validate_sequence(sequence)
    return [cleaned[i : i + 3] for i in range(frame, len(cleaned) - 2, 3)]

def translate_rna(rna: str, frame: int = 0, stop_at_stop: bool = True) -> str:
    cleaned = validate_sequence(rna)
    if detect_sequence_type(cleaned) == "DNA":
        raise ValueError("Expected RNA sequence for RNA translation.")

    amino_acids: list[str] = []
    for codon in split_codons(cleaned, frame):
        amino_acid = STANDARD_RNA_CODON_TABLE.get(codon, "X")
        if amino_acid == "*" and stop_at_stop:
            break
        amino_acids.append(amino_acid)
    return "".join(amino_acids)

def translate_dna(dna: str, frame: int = 0, stop_at_stop: bool = True) -> str:
    return translate_rna(dna_to_rna(dna), frame=frame, stop_at_stop=stop_at_stop)

def find_orfs(dna_or_rna: str, min_length: int = 30) -> list[dict]:
    if min_length < 0:
        raise ValueError("min_length must be non-negative.")

    cleaned = validate_sequence(dna_or_rna)
    sequence_type = detect_sequence_type(cleaned)
    is_rna = sequence_type == "RNA"
    start_codon = "AUG" if is_rna else "ATG"
    stop_codons = {"UAA", "UAG", "UGA"} if is_rna else {"TAA", "TAG", "TGA"}
    orfs: list[dict] = []

    for frame in range(3):
        codon_positions = range(frame, len(cleaned) - 2, 3)
        active_start: int | None = None

        for position in codon_positions:
            codon = cleaned[position : position + 3]
            if active_start is None:
                if codon == start_codon:
                    active_start = position
                continue

            if codon in stop_codons:
                end = position + 3
                sequence = cleaned[active_start:end]
                if len(sequence) >= min_length:
                    protein = (
                        translate_rna(sequence, stop_at_stop=True)
                        if is_rna
                        else translate_dna(sequence, stop_at_stop=True)
                    )
                    orfs.append(
                        {
                            "frame": frame,
                            "start": active_start,
                            "end": end,
                            "length": len(sequence),
                            "sequence": sequence,
                            "protein": protein,
                            "stop_codon": codon,
                        }
                    )
                active_start = None

    return orfs