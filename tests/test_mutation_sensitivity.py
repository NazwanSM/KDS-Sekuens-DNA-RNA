from __future__ import annotations

import numpy as np
from dna_rna_classifier.mutation_sensitivity import (
    compute_promoter_robustness_score,
    format_sensitivity_interpretation,
    generate_single_point_mutations,
    get_changed_kmers,
    get_promoter_score,
    summarize_position_sensitivity,
)

class MockVectorizer:
    """Minimal vectorizer mock for model-dependent tests."""

    def transform(self, sequences: list[str]) -> list[str]:
        """Return sequences unchanged."""
        return sequences

class MockProbabilityModel:
    """Minimal probability model mock."""

    classes_ = np.array([0, 1])

    def predict(self, features: list[str]) -> np.ndarray:
        """Predict promoter when sequence contains more G/C than A/T."""
        return np.array([1 if features[0].count("G") + features[0].count("C") >= 2 else 0])

    def predict_proba(self, features: list[str]) -> np.ndarray:
        """Return a deterministic pseudo-probability."""
        probability = min(0.95, 0.2 + 0.2 * (features[0].count("G") + features[0].count("C")))
        return np.array([[1 - probability, probability]])


def test_generate_single_point_mutations_atg() -> None:
    """ATG produces three alternatives per position and 1-based labels."""
    mutations = generate_single_point_mutations("ATG")
    assert len(mutations) == 9
    assert all(mutation["original_base"] != mutation["mutant_base"] for mutation in mutations)
    assert {mutation["mutation_label"] for mutation in mutations if mutation["position"] == 0} == {
        "A1C",
        "A1G",
        "A1T",
    }

def test_get_changed_kmers_boundaries() -> None:
    """Affected k-mers are exactly windows containing the mutated position."""
    result = get_changed_kmers("ATGCGT", "ATACGT", position=2, k=3)
    assert result["changed_pairs"] == [
        {"start": 0, "end": 3, "original_kmer": "ATG", "mutated_kmer": "ATA"},
        {"start": 1, "end": 4, "original_kmer": "TGC", "mutated_kmer": "TAC"},
        {"start": 2, "end": 5, "original_kmer": "GCG", "mutated_kmer": "ACG"},
    ]

def test_get_promoter_score_with_probability_model() -> None:
    """Probability models expose class-1 promoter probability."""
    score = get_promoter_score(MockProbabilityModel(), MockVectorizer(), "GCGT")
    assert score["predicted_label"] == 1
    assert score["score_type"] == "probability"
    assert score["promoter_probability"] == score["promoter_score"]

def test_summarize_position_sensitivity() -> None:
    """Position-level aggregation computes max/mean drops and most disruptive mutation."""
    results = [
        {
            "position": 0,
            "position_1based": 1,
            "original_base": "A",
            "mutation_label": "A1C",
            "probability_drop": 0.2,
            "absolute_delta_score": 0.2,
        },
        {
            "position": 0,
            "position_1based": 1,
            "original_base": "A",
            "mutation_label": "A1G",
            "probability_drop": 0.4,
            "absolute_delta_score": 0.4,
        },
        {
            "position": 1,
            "position_1based": 2,
            "original_base": "T",
            "mutation_label": "T2A",
            "probability_drop": 0.1,
            "absolute_delta_score": 0.1,
        },
    ]
    summary = summarize_position_sensitivity(results)
    assert summary[0]["position"] == 0
    assert round(summary[0]["mean_probability_drop"], 4) == 0.3
    assert summary[0]["max_probability_drop"] == 0.4
    assert summary[0]["most_disruptive_mutation_label"] == "A1G"

def test_compute_promoter_robustness_score() -> None:
    """Robustness summary computes mean/max drops and threshold fractions."""
    results = [
        {
            "original_predicted_label": 1,
            "mutant_predicted_label": 1,
            "probability_drop": 0.05,
            "delta_score": -0.05,
        },
        {
            "original_predicted_label": 1,
            "mutant_predicted_label": 0,
            "probability_drop": 0.20,
            "delta_score": -0.20,
        },
        {
            "original_predicted_label": 1,
            "mutant_predicted_label": 0,
            "probability_drop": 0.30,
            "delta_score": -0.30,
        },
    ]
    summary = compute_promoter_robustness_score(results)
    assert round(summary["mean_probability_drop"], 4) == 0.1833
    assert summary["max_probability_drop"] == 0.30
    assert summary["fraction_mutations_drop_above_0_10"] == 2 / 3
    assert summary["fraction_mutations_drop_above_0_25"] == 1 / 3
    assert summary["fraction_mutations_flipping_prediction"] == 2 / 3

def test_format_sensitivity_interpretation_cautious() -> None:
    """Interpretation uses cautious model-sensitive wording."""
    scan_result = {
        "original_prediction": {
            "predicted_label": 1,
            "promoter_probability": 0.91,
            "promoter_score": 0.91,
        },
        "top_disruptive_mutations": [
            {
                "mutation_label": "A37G",
                "mutant_probability": 0.42,
                "probability_drop": 0.49,
                "delta_score": -0.49,
            }
        ],
        "position_sensitivity": [
            {"position_1based": 37},
            {"position_1based": 38},
        ],
    }
    text = format_sensitivity_interpretation(scan_result)
    lowered = text.lower()
    assert "model-sensitive" in lowered
    assert "experimentally validated" not in lowered
    assert "proves" not in lowered
    assert "guarantees" not in lowered
