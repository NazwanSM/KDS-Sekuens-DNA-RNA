import pytest
from dna_rna_classifier.validation import clean_sequence, detect_sequence_type, validate_sequence

def test_clean_sequence_lowercase_and_whitespace() -> None:
    """Lowercase and whitespace are normalized."""
    assert clean_sequence(" atg c\n ta\t") == "ATGCTA"

def test_valid_dna_accepted() -> None:
    """Valid DNA is accepted and detected."""
    assert validate_sequence("ATGCN") == "ATGCN"
    assert detect_sequence_type("ATGCN") == "DNA"

def test_valid_rna_accepted() -> None:
    """Valid RNA is accepted and detected."""
    assert validate_sequence("AUGCN") == "AUGCN"
    assert detect_sequence_type("AUGCN") == "RNA"

def test_invalid_characters_rejected() -> None:
    """Invalid symbols raise a helpful validation error."""
    with pytest.raises(ValueError):
        validate_sequence("ATGBZ")