from dna_rna_classifier.central_dogma import (
    find_orfs,
    get_reverse_complement,
    transcribe,
    translate_rna,
)

def test_transcription() -> None:
    """DNA transcription replaces T with U."""
    assert transcribe("ATGGAATAA") == "AUGGAAUAA"

def test_translation_stop_at_stop() -> None:
    """Translation stops before the first stop codon by default."""
    assert translate_rna("AUGGAAUAA", stop_at_stop=True) == "ME"

def test_reverse_complement() -> None:
    """Reverse complement works for DNA."""
    assert get_reverse_complement("ATGC") == "GCAT"

def test_orf_detection() -> None:
    """ORF detection finds a start-to-stop ORF."""
    orfs = find_orfs("CCCATGGAATAACCC", min_length=9)
    assert len(orfs) == 1
    assert orfs[0]["sequence"] == "ATGGAATAA"
    assert orfs[0]["protein"] == "ME"
    assert orfs[0]["stop_codon"] == "TAA"

