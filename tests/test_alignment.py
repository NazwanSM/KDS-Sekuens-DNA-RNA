from dna_rna_classifier.alignment import alignment_metrics, needleman_wunsch

def test_identical_sequences_identity() -> None:
    result = needleman_wunsch("ATGC", "ATGC")
    assert result["metrics"]["identity_percentage"] == 100.0
    assert result["metrics"]["matches"] == 4

def test_simple_mismatch_counted() -> None:
    result = needleman_wunsch("ATGC", "ATGT")
    assert result["metrics"]["mismatches"] == 1
    assert result["metrics"]["gaps"] == 0

def test_gap_counted() -> None:
    result = needleman_wunsch("ATGC", "ATGGC")
    assert result["metrics"]["gaps"] == 1

def test_alignment_metrics_manual() -> None:
    metrics = alignment_metrics("AT-GC", "ATGGC")
    assert metrics["matches"] == 4
    assert metrics["gaps"] == 1

