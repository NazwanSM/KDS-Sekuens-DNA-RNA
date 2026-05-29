from dna_rna_classifier.modeling import KmerTokenizer

def test_kmer_tokenizer_is_callable() -> None:
    tokenizer = KmerTokenizer(k=2)
    assert tokenizer("ATGC") == ["AT", "TG", "GC"]