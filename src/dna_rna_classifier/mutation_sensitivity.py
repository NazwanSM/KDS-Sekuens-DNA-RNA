from __future__ import annotations
from statistics import mean
from typing import Any
from .validation import clean_sequence

DNA_ALPHABET = set("ACGTN")


def generate_single_point_mutations(sequence: str, alphabet: str = "ACGT") -> list[dict]:
    cleaned = _validate_dna_sequence(sequence)
    mutation_alphabet = _validate_mutation_alphabet(alphabet)
    mutations: list[dict] = []

    for position, original_base in enumerate(cleaned):
        for mutant_base in mutation_alphabet:
            if mutant_base == original_base:
                continue
            mutated_sequence = f"{cleaned[:position]}{mutant_base}{cleaned[position + 1:]}"
            mutations.append(
                {
                    "position": position,
                    "position_1based": position + 1,
                    "original_base": original_base,
                    "mutant_base": mutant_base,
                    "mutated_sequence": mutated_sequence,
                    "mutation_label": f"{original_base}{position + 1}{mutant_base}",
                }
            )
    return mutations


def get_promoter_score(model: Any, vectorizer: Any, sequence: str) -> dict:
    cleaned = _validate_dna_sequence(sequence)
    features = vectorizer.transform([cleaned])
    predicted_label = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_index = _class_index(model, 1)
        promoter_probability = float(probabilities[class_index])
        return {
            "predicted_label": predicted_label,
            "promoter_score": promoter_probability,
            "promoter_probability": promoter_probability,
            "score_type": "probability",
        }

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(features)
        if getattr(raw_scores, "ndim", 1) == 1:
            promoter_score = float(raw_scores[0])
        else:
            promoter_score = float(raw_scores[0][_class_index(model, 1)])
        return {
            "predicted_label": predicted_label,
            "promoter_score": promoter_score,
            "promoter_probability": None,
            "score_type": "decision_function",
        }

    raise ValueError("Model must provide predict_proba or decision_function for sensitivity analysis.")


def get_changed_kmers(
    original_sequence: str,
    mutated_sequence: str,
    position: int,
    k: int,
) -> dict:
    original = _validate_dna_sequence(original_sequence)
    mutated = _validate_dna_sequence(mutated_sequence)
    if len(original) != len(mutated):
        raise ValueError("Original and mutated sequences must have the same length.")
    if position < 0 or position >= len(original):
        raise ValueError("position is outside the sequence.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(original) < k:
        return {"original_kmers": [], "mutated_kmers": [], "changed_pairs": []}

    min_start = max(0, position - k + 1)
    max_start = min(position, len(original) - k)
    changed_pairs: list[dict] = []
    original_kmers: list[str] = []
    mutated_kmers: list[str] = []

    for start in range(min_start, max_start + 1):
        end = start + k
        original_kmer = original[start:end]
        mutated_kmer = mutated[start:end]
        original_kmers.append(original_kmer)
        mutated_kmers.append(mutated_kmer)
        changed_pairs.append(
            {
                "start": start,
                "end": end,
                "original_kmer": original_kmer,
                "mutated_kmer": mutated_kmer,
            }
        )

    return {
        "original_kmers": original_kmers,
        "mutated_kmers": mutated_kmers,
        "changed_pairs": changed_pairs,
    }


def scan_mutation_sensitivity(
    sequence: str,
    model: Any,
    vectorizer: Any,
    k: int = 6,
    max_mutations: int | None = None,
) -> dict:
    cleaned = _validate_dna_sequence(sequence)
    if k <= 0:
        raise ValueError("k must be positive.")
    if max_mutations is not None and max_mutations <= 0:
        raise ValueError("max_mutations must be positive when provided.")

    original_prediction = get_promoter_score(model, vectorizer, cleaned)
    original_score = float(original_prediction["promoter_score"])
    original_probability = original_prediction.get("promoter_probability")
    mutations = generate_single_point_mutations(cleaned)
    if max_mutations is not None:
        mutations = mutations[:max_mutations]

    mutation_results: list[dict] = []
    for mutation in mutations:
        mutant_prediction = get_promoter_score(model, vectorizer, mutation["mutated_sequence"])
        mutant_score = float(mutant_prediction["promoter_score"])
        mutant_probability = mutant_prediction.get("promoter_probability")
        delta_score = mutant_score - original_score
        probability_drop = (
            float(original_probability) - float(mutant_probability)
            if original_probability is not None and mutant_probability is not None
            else None
        )
        mutation_results.append(
            {
                "position": mutation["position"],
                "position_1based": mutation["position_1based"],
                "original_base": mutation["original_base"],
                "mutant_base": mutation["mutant_base"],
                "mutation_label": mutation["mutation_label"],
                "original_predicted_label": original_prediction["predicted_label"],
                "original_score": original_score,
                "mutant_score": mutant_score,
                "delta_score": delta_score,
                "absolute_delta_score": abs(delta_score),
                "original_probability": original_probability,
                "mutant_probability": mutant_probability,
                "probability_drop": probability_drop,
                "mutant_predicted_label": mutant_prediction["predicted_label"],
                "changed_kmers": get_changed_kmers(
                    cleaned,
                    mutation["mutated_sequence"],
                    mutation["position"],
                    k,
                ),
            }
        )

    if original_probability is not None:
        disruptive = sorted(
            mutation_results,
            key=lambda result: _none_to_negative_infinity(result["probability_drop"]),
            reverse=True,
        )
        neutral = sorted(
            mutation_results,
            key=lambda result: abs(float(result["probability_drop"] or 0.0)),
        )
    else:
        disruptive = sorted(mutation_results, key=lambda result: result["delta_score"])
        neutral = sorted(mutation_results, key=lambda result: result["absolute_delta_score"])

    return {
        "original_prediction": original_prediction,
        "mutation_results": mutation_results,
        "top_disruptive_mutations": disruptive[:10],
        "top_neutral_mutations": neutral[:10],
        "position_sensitivity": summarize_position_sensitivity(mutation_results),
        "robustness_summary": compute_promoter_robustness_score(mutation_results),
    }


def summarize_position_sensitivity(mutation_results: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for result in mutation_results:
        grouped.setdefault(int(result["position"]), []).append(result)

    summaries: list[dict] = []
    for position, results in grouped.items():
        probability_drops = [
            float(result["probability_drop"])
            for result in results
            if result.get("probability_drop") is not None
        ]
        abs_deltas = [float(result["absolute_delta_score"]) for result in results]
        if probability_drops:
            most_disruptive = max(results, key=lambda result: float(result.get("probability_drop") or 0.0))
        else:
            most_disruptive = max(results, key=lambda result: float(result["absolute_delta_score"]))
        summaries.append(
            {
                "position": position,
                "position_1based": int(results[0]["position_1based"]),
                "original_base": str(results[0]["original_base"]),
                "n_mutations": len(results),
                "mean_probability_drop": mean(probability_drops) if probability_drops else None,
                "max_probability_drop": max(probability_drops) if probability_drops else None,
                "min_probability_drop": min(probability_drops) if probability_drops else None,
                "mean_abs_delta_score": mean(abs_deltas) if abs_deltas else 0.0,
                "most_disruptive_mutation_label": most_disruptive["mutation_label"],
            }
        )

    if any(summary["max_probability_drop"] is not None for summary in summaries):
        return sorted(
            summaries,
            key=lambda summary: _none_to_negative_infinity(summary["max_probability_drop"]),
            reverse=True,
        )
    return sorted(summaries, key=lambda summary: summary["mean_abs_delta_score"], reverse=True)


def compute_promoter_robustness_score(mutation_results: list[dict]) -> dict:
    if not mutation_results:
        return {
            "n_mutations": 0,
            "score_type": "unknown",
            "mean_probability_drop": None,
            "max_probability_drop": None,
            "fraction_mutations_drop_above_0_10": None,
            "fraction_mutations_drop_above_0_25": None,
            "fraction_mutations_flipping_prediction": 0.0,
        }

    n_mutations = len(mutation_results)
    original_label = int(mutation_results[0].get("original_predicted_label", -1))
    if original_label == -1:
        original_label = _infer_original_label(mutation_results)
    flip_fraction = sum(
        1 for result in mutation_results if int(result.get("mutant_predicted_label", original_label)) != original_label
    ) / n_mutations

    probability_drops = [
        float(result["probability_drop"])
        for result in mutation_results
        if result.get("probability_drop") is not None
    ]
    if probability_drops:
        return {
            "n_mutations": n_mutations,
            "score_type": "probability",
            "mean_probability_drop": mean(probability_drops),
            "max_probability_drop": max(probability_drops),
            "fraction_mutations_drop_above_0_10": _fraction_above(probability_drops, 0.10),
            "fraction_mutations_drop_above_0_25": _fraction_above(probability_drops, 0.25),
            "fraction_mutations_flipping_prediction": flip_fraction,
        }

    delta_scores = [float(result["delta_score"]) for result in mutation_results]
    negative_drops = [-delta for delta in delta_scores]
    return {
        "n_mutations": n_mutations,
        "score_type": "decision_function",
        "mean_score_decrease": mean(negative_drops),
        "max_score_decrease": max(negative_drops),
        "fraction_mutations_score_decrease_above_0_10": _fraction_above(negative_drops, 0.10),
        "fraction_mutations_score_decrease_above_0_25": _fraction_above(negative_drops, 0.25),
        "fraction_mutations_flipping_prediction": flip_fraction,
    }


def format_sensitivity_interpretation(scan_result: dict) -> str:
    original = scan_result.get("original_prediction", {})
    top_disruptive = scan_result.get("top_disruptive_mutations", [])
    positions = scan_result.get("position_sensitivity", [])
    predicted_label = int(original.get("predicted_label", -1))
    probability = original.get("promoter_probability")

    if probability is not None:
        base_sentence = (
            f"The original sequence is predicted as "
            f"{'promoter' if predicted_label == 1 else 'non-promoter'} "
            f"with promoter probability {float(probability):.3f}."
        )
    else:
        base_sentence = (
            f"The original sequence is predicted as "
            f"{'promoter' if predicted_label == 1 else 'non-promoter'} "
            "using the model decision score."
        )

    if top_disruptive:
        top = top_disruptive[0]
        if top.get("mutant_probability") is not None:
            mutation_sentence = (
                f"The most disruptive in-silico mutation is {top['mutation_label']}, "
                f"changing promoter probability to {float(top['mutant_probability']):.3f} "
                f"(drop {float(top['probability_drop']):.3f})."
            )
        else:
            mutation_sentence = (
                f"The most disruptive in-silico mutation is {top['mutation_label']}, "
                f"with model-score change {float(top['delta_score']):.3f}."
            )
    else:
        mutation_sentence = "No in-silico point mutations were scanned."

    if positions:
        top_positions = sorted(position["position_1based"] for position in positions[:5])
        region_sentence = (
            f"Positions {', '.join(str(position) for position in top_positions)} show higher "
            "model sensitivity, indicating that local k-mer patterns in these regions strongly "
            "affect the classifier prediction."
        )
    else:
        region_sentence = "No position-level sensitivity summary is available."

    caution = (
        "These positions are model-sensitive regions, not verified promoter motifs "
        "or clinically actionable variants."
    )
    if predicted_label == 0:
        caution = (
            "Since the original sequence is predicted as non-promoter, mutation sensitivity should be "
            "interpreted as changes in model score rather than loss of promoter function. "
            + caution
        )

    return " ".join([base_sentence, mutation_sentence, region_sentence, caution])


def _validate_dna_sequence(sequence: str) -> str:
    cleaned = clean_sequence(sequence)
    if not cleaned:
        raise ValueError("DNA sequence must be non-empty.")
    if not set(cleaned).issubset(DNA_ALPHABET):
        raise ValueError("DNA sequence must contain only A, C, G, T, and N.")
    return cleaned


def _validate_mutation_alphabet(alphabet: str) -> list[str]:
    cleaned = clean_sequence(alphabet).replace("N", "")
    if not cleaned:
        raise ValueError("Mutation alphabet must contain at least one A/C/G/T base.")
    if not set(cleaned).issubset(set("ACGT")):
        raise ValueError("Mutation alphabet must contain only A, C, G, and T.")
    return list(dict.fromkeys(cleaned))


def _class_index(model: Any, class_label: int) -> int:
    classes = [int(label) for label in getattr(model, "classes_", [])]
    if class_label not in classes:
        raise ValueError(f"Model classes must include class {class_label}. Found: {classes}")
    return classes.index(class_label)


def _fraction_above(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value > threshold) / len(values)


def _none_to_negative_infinity(value: float | None) -> float:
    return float(value) if value is not None else float("-inf")


def _infer_original_label(mutation_results: list[dict]) -> int:
    first = mutation_results[0]
    probability = first.get("original_probability")
    if probability is not None:
        return 1 if float(probability) >= 0.5 else 0
    return 1 if float(first.get("original_score", 0.0)) >= 0 else 0
