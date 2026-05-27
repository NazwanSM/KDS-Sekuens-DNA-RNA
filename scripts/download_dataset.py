from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dna_rna_classifier.promoter_dataset import (  # noqa: E402
    DATASET_NAME,
    TASK_NAME,
    validate_promoter_dataframe,
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the strict dataset downloader."""
    default_cache_dir = str(ROOT / ".cache" / "huggingface")
    if os.name == "nt":
        default_cache_dir = str(Path.home().drive + "\\hf_cache")
    parser = argparse.ArgumentParser(description="Download real promoter_all train/test CSV files.")
    parser.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset name.")
    parser.add_argument("--task", default=TASK_NAME, help="Task to filter, expected promoter_all.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for exported CSV files.")
    parser.add_argument(
        "--hf-cache-dir",
        default=default_cache_dir,
        help="Local Hugging Face cache directory.",
    )
    return parser.parse_args()

def _load_dataset_dict(dataset_name: str, cache_dir: str) -> object:
    """Load the Hugging Face dataset dictionary or raise a clear error."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_dir) / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(cache_dir) / "datasets"))
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        return load_dataset(dataset_name, cache_dir=cache_dir)
    except Exception as exc:
        import traceback

        raise RuntimeError(
            f"Could not load real dataset '{dataset_name}'. "
            "Check your internet connection and Hugging Face access. "
            f"Original error: {exc}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        ) from exc

def _export_split(dataset_dict: object, split: str, task: str, output_path: Path) -> int:
    """Filter one official split by task, validate it, and write CSV."""
    if split not in dataset_dict:
        raise ValueError(f"Dataset must provide an official '{split}' split.")

    dataset_split = dataset_dict[split]
    if "task" not in dataset_split.column_names:
        raise ValueError(f"Split '{split}' is missing required column 'task'.")

    filtered = dataset_split.filter(lambda example: example["task"] == task)
    if len(filtered) == 0:
        raise ValueError(f"Split '{split}' has no rows for task '{task}'.")

    validated = validate_promoter_dataframe(filtered.to_pandas(), task=task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_path, index=False)
    return len(validated)

def main() -> int:
    """Download and export promoter_all train/test CSV files."""
    args = parse_args()
    if args.dataset != DATASET_NAME:
        raise ValueError(f"This project is configured for dataset '{DATASET_NAME}', got '{args.dataset}'.")
    if args.task != TASK_NAME:
        raise ValueError(f"This project is configured for task '{TASK_NAME}', got '{args.task}'.")

    output_dir = Path(args.output_dir)
    dataset_dict = _load_dataset_dict(args.dataset, args.hf_cache_dir)

    train_path = output_dir / f"{args.task}_train.csv"
    test_path = output_dir / f"{args.task}_test.csv"
    train_rows = _export_split(dataset_dict, "train", args.task, train_path)
    test_rows = _export_split(dataset_dict, "test", args.task, test_path)

    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Wrote {train_rows} train rows to {train_path}")
    print(f"Wrote {test_rows} test rows to {test_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())