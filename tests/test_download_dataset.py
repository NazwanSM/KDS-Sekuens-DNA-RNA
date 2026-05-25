from __future__ import annotations
import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

class FakeDatasetSplit:
    """Small stand-in for a Hugging Face dataset split."""

    def __init__(self, rows: list[dict]) -> None:
        """Store rows for filtering and dataframe conversion."""
        self._rows = rows
        self.column_names = list(rows[0].keys())

    def filter(self, predicate: object) -> "FakeDatasetSplit":
        """Return rows accepted by a Hugging Face-style predicate."""
        return FakeDatasetSplit([row for row in self._rows if predicate(row)])

    def to_pandas(self) -> pd.DataFrame:
        """Return rows as a dataframe."""
        return pd.DataFrame(self._rows)

    def __len__(self) -> int:
        """Return row count."""
        return len(self._rows)

def _load_download_module() -> object:
    """Load scripts/download_dataset.py as an importable module."""
    script_path = Path("scripts/download_dataset.py")
    spec = importlib.util.spec_from_file_location("download_dataset_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_download_dataset_writes_required_promoter_csvs(monkeypatch: object) -> None:
    """Downloader exports non-empty train/test CSV files with strict promoter_all schema."""
    module = _load_download_module()
    output_dir = Path(".tmp/test_download_dataset")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    rows = {
        "train": [
            {"sequence": "ACGTN", "label": 0, "task": "promoter_all", "name": "fixture_train_0"},
            {"sequence": "TGCAN", "label": 1, "task": "promoter_all", "name": "fixture_train_1"},
            {"sequence": "ACGTN", "label": 1, "task": "other_task", "name": "ignored_train"},
        ],
        "test": [
            {"sequence": "NNACGT", "label": 0, "task": "promoter_all", "name": "fixture_test_0"},
            {"sequence": "TTACGN", "label": 1, "task": "promoter_all", "name": "fixture_test_1"},
        ],
    }

    def fake_load_dataset(dataset_name: str, cache_dir: str) -> dict:
        return {
            "train": FakeDatasetSplit(rows["train"]),
            "test": FakeDatasetSplit(rows["test"]),
        }

    fake_datasets_module = SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_dataset.py",
            "--dataset",
            "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
            "--task",
            "promoter_all",
            "--output-dir",
            str(output_dir),
        ],
    )

    try:
        assert module.main() == 0
        train_csv = output_dir / "promoter_all_train.csv"
        test_csv = output_dir / "promoter_all_test.csv"
        assert train_csv.exists()
        assert test_csv.exists()

        for csv_path in [train_csv, test_csv]:
            df = pd.read_csv(csv_path)
            assert list(df.columns) == ["sequence", "label", "task", "name"]
            assert not df.empty
            assert set(df["task"]) == {"promoter_all"}
            assert set(df["label"]).issubset({0, 1})
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)