from __future__ import annotations
from .alignment import format_alignment, needleman_wunsch
from .central_dogma import translate_dna, translate_rna
from .validation import detect_sequence_type, validate_sequence

def _to_rna_codon(codon: str) -> str:
    return validate_sequence(codon).replace("T", "U")

def _translate_sequence(sequence: str, frame: int = 0) -> str:
    sequence_type = detect_sequence_type(sequence)
    if sequence_type == "RNA":
        return translate_rna(sequence, frame=frame, stop_at_stop=True)
    return translate_dna(sequence, frame=frame, stop_at_stop=True)

def compare_sequences(reference: str, mutated: str) -> list[dict]:
    clean_ref = validate_sequence(reference)
    clean_mut = validate_sequence(mutated)
    alignment = needleman_wunsch(clean_ref, clean_mut)
    aligned_ref = alignment["aligned_seq1"]
    aligned_mut = alignment["aligned_seq2"]

    ref_pos = 0
    mut_pos = 0
    mutations: list[dict] = []
    for ref_base, mut_base in zip(aligned_ref, aligned_mut):
        if ref_base != "-":
            ref_pos += 1
        if mut_base != "-":
            mut_pos += 1

        if ref_base == mut_base:
            continue
        if ref_base == "-":
            mutations.append(
                {
                    "type": "insertion",
                    "position": ref_pos + 1,
                    "reference_position": ref_pos,
                    "mutated_position": mut_pos,
                    "reference": "-",
                    "mutated": mut_base,
                }
            )
        elif mut_base == "-":
            mutations.append(
                {
                    "type": "deletion",
                    "position": ref_pos,
                    "reference_position": ref_pos,
                    "mutated_position": mut_pos + 1,
                    "reference": ref_base,
                    "mutated": "-",
                }
            )
        else:
            mutations.append(
                {
                    "type": "substitution",
                    "position": ref_pos,
                    "reference_position": ref_pos,
                    "mutated_position": mut_pos,
                    "reference": ref_base,
                    "mutated": mut_base,
                }
            )
    return mutations

def classify_point_mutation(reference_codon: str, mutated_codon: str) -> dict:
    ref_rna = _to_rna_codon(reference_codon)
    mut_rna = _to_rna_codon(mutated_codon)
    if len(ref_rna) != 3 or len(mut_rna) != 3:
        raise ValueError("Point mutation classification requires codons of length 3.")

    ref_aa = translate_rna(ref_rna, stop_at_stop=False)
    mut_aa = translate_rna(mut_rna, stop_at_stop=False)
    if ref_aa == "X" or mut_aa == "X":
        mutation_type = "unknown"
    elif ref_aa == mut_aa:
        mutation_type = "silent"
    elif mut_aa == "*" and ref_aa != "*":
        mutation_type = "nonsense"
    else:
        mutation_type = "missense"

    return {
        "reference_codon": reference_codon,
        "mutated_codon": mutated_codon,
        "reference_amino_acid": ref_aa,
        "mutated_amino_acid": mut_aa,
        "mutation_type": mutation_type,
    }

def analyze_mutations(reference: str, mutated: str, frame: int = 0) -> dict:
    if frame not in {0, 1, 2}:
        raise ValueError("Reading frame must be 0, 1, or 2.")

    clean_ref = validate_sequence(reference)
    clean_mut = validate_sequence(mutated)
    mutations = compare_sequences(clean_ref, clean_mut)
    length_difference = len(clean_mut) - len(clean_ref)
    frameshift = length_difference % 3 != 0

    ref_protein = _translate_sequence(clean_ref, frame=frame)
    mut_protein = _translate_sequence(clean_mut, frame=frame)
    codon_effects: list[dict] = []
    if not frameshift:
        max_codon_end = min(len(clean_ref), len(clean_mut)) - 2
        for start in range(frame, max_codon_end, 3):
            ref_codon = clean_ref[start : start + 3]
            mut_codon = clean_mut[start : start + 3]
            if len(ref_codon) == 3 and len(mut_codon) == 3 and ref_codon != mut_codon:
                effect = classify_point_mutation(ref_codon, mut_codon)
                effect["codon_start"] = start
                effect["codon_end"] = start + 3
                codon_effects.append(effect)

    alignment = needleman_wunsch(clean_ref, clean_mut)
    mutation_counts: dict[str, int] = {}
    for mutation in mutations:
        mutation_counts[mutation["type"]] = mutation_counts.get(mutation["type"], 0) + 1

    if not mutations:
        summary = "No sequence-level mutations were detected."
    else:
        parts = [f"{count} {name}" for name, count in sorted(mutation_counts.items())]
        summary = "Detected " + ", ".join(parts) + "."
        if frameshift:
            summary += " The length difference is not divisible by 3, so this is likely a frameshift mutation."
        elif codon_effects:
            effect_names = sorted({effect["mutation_type"] for effect in codon_effects})
            summary += " Codon-level effects: " + ", ".join(effect_names) + "."
        if ref_protein != mut_protein:
            summary += " The translated protein sequence changes."
        else:
            summary += " The translated protein sequence is unchanged in the selected frame."

    return {
        "reference_sequence": clean_ref,
        "mutated_sequence": clean_mut,
        "reference_type": detect_sequence_type(clean_ref),
        "mutated_type": detect_sequence_type(clean_mut),
        "mutations": mutations,
        "mutation_count": len(mutations),
        "mutation_counts": mutation_counts,
        "frameshift": frameshift,
        "length_difference": length_difference,
        "reference_protein": ref_protein,
        "mutated_protein": mut_protein,
        "protein_changed": ref_protein != mut_protein,
        "codon_effects": codon_effects,
        "summary": summary,
        "alignment": {
            **alignment,
            "formatted": format_alignment(alignment["aligned_seq1"], alignment["aligned_seq2"]),
        },
    }