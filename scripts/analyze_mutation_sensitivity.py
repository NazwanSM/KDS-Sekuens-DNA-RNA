from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dna_rna_classifier.mutation_sensitivity import (
    format_sensitivity_interpretation,
    scan_mutation_sensitivity,
)
from dna_rna_classifier.promoter_dataset import TASK_NAME, read_promoter_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze promoter mutation sensitivity.")
    parser.add_argument("--sequence", default=None, help="Optional manual DNA sequence.")
    parser.add_argument("--test-csv", default="data/processed/promoter_all_test.csv")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--only-promoters", action="store_true")
    parser.add_argument("--model-path", default="models/promoter_kmer_logreg.joblib")
    parser.add_argument("--vectorizer-path", default="models/promoter_kmer_vectorizer.joblib")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--output-dir", default="reports/mutation_sensitivity")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_model_and_vectorizer(model_path: Path, vectorizer_path: Path) -> tuple[Any, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run scripts/train_baseline.py first.")
    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {vectorizer_path}. Run scripts/train_baseline.py first."
        )
    return joblib.load(model_path), joblib.load(vectorizer_path)


def _load_sequences(args: argparse.Namespace) -> list[dict]:
    if args.sequence:
        return [{"sequence_name": "manual_sequence", "true_label": None, "sequence": args.sequence}]

    test_csv = Path(args.test_csv)
    test_df = read_promoter_csv(test_csv, task=TASK_NAME)
    if args.only_promoters:
        test_df = test_df[test_df["label"] == 1].copy()
    if test_df.empty:
        raise ValueError("No real test sequences are available after filtering.")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    sample_size = min(args.sample_size, len(test_df))
    sampled = test_df.sample(sample_size, random_state=args.random_state)
    return [
        {
            "sequence_name": str(row["name"]),
            "true_label": int(row["label"]),
            "sequence": str(row["sequence"]),
        }
        for _, row in sampled.iterrows()
    ]


def _flatten_mutation_results(sequence_info: dict, scan_result: dict) -> list[dict]:
    rows: list[dict] = []
    for result in scan_result["mutation_results"]:
        rows.append(
            {
                "sequence_name": sequence_info["sequence_name"],
                "true_label": sequence_info["true_label"],
                "mutation_label": result["mutation_label"],
                "position": result["position"],
                "position_1based": result["position_1based"],
                "original_base": result["original_base"],
                "mutant_base": result["mutant_base"],
                "original_score": result["original_score"],
                "mutant_score": result["mutant_score"],
                "delta_score": result["delta_score"],
                "absolute_delta_score": result["absolute_delta_score"],
                "original_probability": result["original_probability"],
                "mutant_probability": result["mutant_probability"],
                "probability_drop": result["probability_drop"],
                "original_predicted_label": result["original_predicted_label"],
                "mutant_predicted_label": result["mutant_predicted_label"],
                "changed_kmers": json.dumps(result["changed_kmers"]["changed_pairs"]),
            }
        )
    return rows


def _flatten_position_results(sequence_info: dict, scan_result: dict) -> list[dict]:
    return [
        {
            "sequence_name": sequence_info["sequence_name"],
            "true_label": sequence_info["true_label"],
            **position_result,
        }
        for position_result in scan_result["position_sensitivity"]
    ]


def _plot_top_disruptive(results_df: pd.DataFrame, output_path: Path) -> None:
    if results_df.empty:
        return
    metric = "probability_drop" if results_df["probability_drop"].notna().any() else "delta_score"
    if metric == "probability_drop":
        plot_df = results_df.sort_values(metric, ascending=False).head(15)
        ylabel = "Probability drop"
    else:
        plot_df = results_df.sort_values(metric, ascending=True).head(15)
        ylabel = "Delta score"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(plot_df["mutation_label"], plot_df[metric])
    ax.set_xlabel("Mutation")
    ax.set_ylabel(ylabel)
    ax.set_title("Top disruptive in-silico point mutations")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_position_sensitivity(position_df: pd.DataFrame, output_path: Path) -> None:
    if position_df.empty:
        return
    metric = (
        "max_probability_drop"
        if position_df["max_probability_drop"].notna().any()
        else "mean_abs_delta_score"
    )
    plot_df = position_df.sort_values("position_1based")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for sequence_name, group in plot_df.groupby("sequence_name"):
        ax.plot(group["position_1based"], group[metric], label=sequence_name, alpha=0.85)
    ax.set_xlabel("Position (1-based)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Position sensitivity across scanned sequences")
    if plot_df["sequence_name"].nunique() <= 5:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive.")

    model, vectorizer = _load_model_and_vectorizer(Path(args.model_path), Path(args.vectorizer_path))
    sequences = _load_sequences(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mutation_rows: list[dict] = []
    position_rows: list[dict] = []
    sequence_summaries: list[dict] = []

    for sequence_info in sequences:
        scan_result = scan_mutation_sensitivity(sequence_info["sequence"], model, vectorizer, k=args.k)
        mutation_rows.extend(_flatten_mutation_results(sequence_info, scan_result))
        position_rows.extend(_flatten_position_results(sequence_info, scan_result))
        sequence_summaries.append(
            {
                "sequence_name": sequence_info["sequence_name"],
                "true_label": sequence_info["true_label"],
                "sequence_length": len(sequence_info["sequence"]),
                "original_prediction": scan_result["original_prediction"],
                "robustness_summary": scan_result["robustness_summary"],
                "top_disruptive_mutation": (
                    scan_result["top_disruptive_mutations"][0]
                    if scan_result["top_disruptive_mutations"]
                    else None
                ),
                "interpretation": format_sensitivity_interpretation(scan_result),
            }
        )

    mutation_df = pd.DataFrame(mutation_rows)
    position_df = pd.DataFrame(position_rows)
    mutation_df.to_csv(output_dir / "sensitivity_results.csv", index=False)
    position_df.to_csv(output_dir / "position_sensitivity.csv", index=False)

    summary = {
        "analysis": "Promoter Mutation Sensitivity Analyzer",
        "k": args.k,
        "model_path": str(Path(args.model_path)),
        "vectorizer_path": str(Path(args.vectorizer_path)),
        "n_sequences": len(sequences),
        "n_mutations_scanned": int(len(mutation_df)),
        "uses_generated_mutants_for_training": False,
        "limitation": (
            "This mutation sensitivity analysis is a computational interpretation of the trained "
            "k-mer model. It identifies model-sensitive positions, not experimentally validated "
            "promoter motifs or clinically actionable variants."
        ),
        "sequence_summaries": sequence_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_top_disruptive(mutation_df, output_dir / "top_disruptive_mutations.png")
    _plot_position_sensitivity(position_df, output_dir / "position_sensitivity.png")

    print(f"Analyzed {len(sequences)} sequence(s).")
    print(f"Scanned {len(mutation_df)} in-silico point mutations.")
    print(f"Saved mutation results to: {output_dir / 'sensitivity_results.csv'}")
    print(f"Saved position sensitivity to: {output_dir / 'position_sensitivity.csv'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

