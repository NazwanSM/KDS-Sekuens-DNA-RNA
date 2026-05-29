from dna_rna_classifier.central_dogma import (
    find_orfs,
    get_reverse_complement,
    transcribe,
    translate_rna,
)

def test_transcription() -> None:
    assert transcribe("ATGGAATAA") == "AUGGAAUAA"

def test_translation_stop_at_stop() -> None:
    assert translate_rna("AUGGAAUAA", stop_at_stop=True) == "ME"

def test_reverse_complement() -> None:
    assert get_reverse_complement("ATGC") == "GCAT"

def test_orf_detection() -> None:
    orfs = find_orfs("CCCATGGAATAACCC", min_length=9)
    assert len(orfs) == 1
    assert orfs[0]["sequence"] == "ATGGAATAA"
    assert orfs[0]["protein"] == "ME"
    assert orfs[0]["stop_codon"] == "TAA"

