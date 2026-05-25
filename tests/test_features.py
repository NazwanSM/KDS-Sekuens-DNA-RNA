from dna_rna_classifier.features import find_motifs, gc_content, kmerize

def test_gc_content_atgc() -> None:
    """ATGC has 50 percent GC content."""
    assert gc_content("ATGC") == 50.0

def test_kmerize_two_mers() -> None:
    """k-merization returns overlapping windows."""
    assert kmerize("ATGC", k=2) == ["AT", "TG", "GC"]

def test_find_motifs_positions() -> None:
    """Motif search reports zero-based start and exclusive end positions."""
    hits = find_motifs("ATATATA", ["TATA"])
    assert hits == [
        {"motif": "TATA", "start": 1, "end": 5, "match": "TATA"},
        {"motif": "TATA", "start": 3, "end": 7, "match": "TATA"},
    ]