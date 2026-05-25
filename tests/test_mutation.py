from dna_rna_classifier.mutation import analyze_mutations, classify_point_mutation

def test_silent_mutation() -> None:
    """GAA to GAG keeps glutamate."""
    assert classify_point_mutation("GAA", "GAG")["mutation_type"] == "silent"

def test_missense_mutation() -> None:
    """GAA to GUA changes glutamate to valine."""
    assert classify_point_mutation("GAA", "GUA")["mutation_type"] == "missense"

def test_nonsense_mutation() -> None:
    """GAA to UAA introduces a stop codon."""
    assert classify_point_mutation("GAA", "UAA")["mutation_type"] == "nonsense"

def test_frameshift_insertion() -> None:
    """Length changes not divisible by three are frameshift candidates."""
    result = analyze_mutations("ATGGAATAA", "ATGGAAATAA")
    assert result["frameshift"] is True
    assert any(mutation["type"] == "insertion" for mutation in result["mutations"])