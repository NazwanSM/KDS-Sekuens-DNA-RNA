# Data

The main training and evaluation dataset is **not stored in the repository**.

Use the real Hugging Face dataset:

- Dataset: `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`
- Task: `promoter_all`
- Objective: binary promoter classification
  - `0 = non-promoter`
  - `1 = promoter`

Download and export the official train/test splits:

```bash
python scripts/download_dataset.py --dataset InstaDeepAI/nucleotide_transformer_downstream_tasks_revised --task promoter_all --output-dir data/processed
```

This writes:

- `data/processed/promoter_all_train.csv`
- `data/processed/promoter_all_test.csv`

Synthetic, random, toy, placeholder, or manually invented sequences must not be used for model training, model evaluation, default demo examples, or reported scientific results. Tiny hardcoded strings may appear only in unit tests for deterministic utility functions such as validation, GC content, k-mers, transcription, translation, alignment, and mutation classification.

