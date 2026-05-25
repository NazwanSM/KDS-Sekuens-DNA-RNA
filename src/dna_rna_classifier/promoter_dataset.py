from __future__ import annotations
from pathlib import Path
import pandas as pd

DATASET_NAME = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
TASK_NAME = "promoter_all"
REQUIRED_COLUMNS = ["sequence", "label", "task", "name"]
LABEL_MAPPING = {0: "non-promoter", 1: "promoter"}
DNA_ALPHABET = set("ACGTN")

def validate_promoter_dataframe(df: pd.DataFrame, task: str = TASK_NAME) -> pd.DataFrame:
    """Validate and standardize promoter_all rows.

    The returned dataframe contains exactly ``sequence``, ``label``, ``task``,
    and ``name`` columns. Invalid data raises ``ValueError`` immediately so the
    pipeline never falls back to misleading synthetic data.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available columns: {list(df.columns)}")

    working = df[REQUIRED_COLUMNS].copy()
    if working.empty:
        raise ValueError(f"No rows available for task '{task}'.")

    working["sequence"] = working["sequence"].map(_normalize_sequence)
    if working["sequence"].isna().any() or (working["sequence"] == "").any():
        raise ValueError("Every sequence must be a non-empty string.")

    invalid_sequence_mask = ~working["sequence"].map(lambda sequence: set(sequence).issubset(DNA_ALPHABET))
    if invalid_sequence_mask.any():
        bad = working.loc[invalid_sequence_mask, "sequence"].head(3).tolist()
        raise ValueError(f"Found sequence(s) with non-DNA characters. Examples: {bad}")

    try:
        labels = working["label"].map(_normalize_label).astype(int)
    except ValueError as exc:
        raise ValueError("Labels must be integer values 0 or 1.") from exc
    if not set(labels.unique()).issubset(set(LABEL_MAPPING)):
        raise ValueError(f"Labels for {task} must be binary 0/1. Found: {sorted(labels.unique())}")
    working["label"] = labels

    if not (working["task"].astype(str) == task).all():
        found = sorted(working["task"].astype(str).unique().tolist())
        raise ValueError(f"Every row must have task == '{task}'. Found: {found}")
    working["task"] = task

    if working["name"].isna().any() or (working["name"].astype(str).str.strip() == "").any():
        raise ValueError("Every row must include a non-empty name.")
    working["name"] = working["name"].astype(str)

    if working.empty:
        raise ValueError(f"Validated dataset for task '{task}' is empty.")
    return working[REQUIRED_COLUMNS]

def read_promoter_csv(path: str | Path, task: str = TASK_NAME) -> pd.DataFrame:
    """Read and validate a promoter_all CSV file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Required real dataset CSV not found: {csv_path}. "
            "Run scripts/download_dataset.py first."
        )
    return validate_promoter_dataframe(pd.read_csv(csv_path), task=task)

def _normalize_sequence(value: object) -> str:
    """Normalize one DNA sequence value."""
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def _normalize_label(value: object) -> int:
    """Normalize one binary label value."""
    if isinstance(value, bool):
        raise ValueError("Boolean labels are not accepted.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid label: {value!r}")

