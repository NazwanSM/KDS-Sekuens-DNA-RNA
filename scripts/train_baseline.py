from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dna_rna_classifier.modeling import KmerTokenizer  # noqa: E402
from dna_rna_classifier.promoter_dataset import (  # noqa: E402
    DATASET_NAME,
    LABEL_MAPPING,
    TASK_NAME,
    read_promoter_csv,
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a promoter_all k-mer baseline model.")
    parser.add_argument("--train-csv", default="data/processed/promoter_all_train.csv")
    parser.add_argument("--model-output", default="models/promoter_kmer_logreg.joblib")
    parser.add_argument("--vectorizer-output", default="models/promoter_kmer_vectorizer.joblib")
    parser.add_argument("--metadata-output", default="models/promoter_kmer_metadata.json")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "linear_svm", "random_forest"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()

def build_model(model_name: str, random_state: int) -> object:
    """Build a supported scikit-learn classifier."""
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_name == "linear_svm":
        return LinearSVC(random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    raise ValueError(f"Unsupported model type: {model_name}")

def main() -> int:
    """Train and persist model, vectorizer, and metadata."""
    args = parse_args()
    if args.k <= 0:
        raise ValueError("k-mer size must be positive.")

    train_df = read_promoter_csv(args.train_csv, task=TASK_NAME)
    vectorizer = CountVectorizer(analyzer=KmerTokenizer(k=args.k), lowercase=False)
    model = build_model(args.model, args.random_state)

    features = vectorizer.fit_transform(train_df["sequence"])
    model.fit(features, train_df["label"])

    model_output = Path(args.model_output)
    vectorizer_output = Path(args.vectorizer_output)
    metadata_output = Path(args.metadata_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_output)
    joblib.dump(vectorizer, vectorizer_output)

    metadata = {
        "dataset_name": DATASET_NAME,
        "task_name": TASK_NAME,
        "training_csv_path": str(Path(args.train_csv)),
        "number_of_training_samples": int(len(train_df)),
        "kmer_size": int(args.k),
        "model_type": args.model,
        "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Dataset: {DATASET_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Training samples: {len(train_df)}")
    print(f"k-mer size: {args.k}")
    print(f"Model type: {args.model}")
    print(f"Saved model to: {model_output}")
    print(f"Saved vectorizer to: {vectorizer_output}")
    print(f"Saved metadata to: {metadata_output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())