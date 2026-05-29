from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dna_rna_classifier.promoter_dataset import DATASET_NAME, TASK_NAME, read_promoter_csv 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a promoter_all k-mer baseline model.")
    parser.add_argument("--test-csv", default="data/processed/promoter_all_test.csv")
    parser.add_argument("--model-path", default="models/promoter_kmer_logreg.joblib")
    parser.add_argument("--vectorizer-path", default="models/promoter_kmer_vectorizer.joblib")
    parser.add_argument("--output-dir", default="reports/evaluation")
    return parser.parse_args()

def _save_confusion_matrix_png(matrix: list[list[int]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["0 non-promoter", "1 promoter"])
    ax.set_yticks([0, 1], labels=["0 non-promoter", "1 promoter"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def main() -> int:
    args = parse_args()
    test_df = read_promoter_csv(args.test_csv, task=TASK_NAME)
    model_path = Path(args.model_path)
    vectorizer_path = Path(args.vectorizer_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run scripts/train_baseline.py first.")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}. Run scripts/train_baseline.py first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    features = vectorizer.transform(test_df["sequence"])
    predictions = model.predict(features)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["label"], predictions, average="binary", pos_label=1, zero_division=0
    )
    matrix = confusion_matrix(test_df["label"], predictions, labels=[0, 1]).tolist()
    report_text = classification_report(
        test_df["label"],
        predictions,
        labels=[0, 1],
        target_names=["non-promoter", "promoter"],
        zero_division=0,
    )

    metrics = {
        "dataset": DATASET_NAME,
        "task": TASK_NAME,
        "test_csv_path": str(Path(args.test_csv)),
        "number_of_test_samples": int(len(test_df)),
        "accuracy": float(accuracy_score(test_df["label"], predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    _save_confusion_matrix_png(matrix, output_dir / "confusion_matrix.png")

    print(f"Dataset: {DATASET_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Test samples: {len(test_df)}")
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        print(f"{key}: {metrics[key]:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")
    print(f"Saved classification report to: {output_dir / 'classification_report.txt'}")
    print(f"Saved confusion matrix image to: {output_dir / 'confusion_matrix.png'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())