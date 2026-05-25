# DNA/RNA Sequence Function Classifier + Mutation Explainer

Project IF3211 Domain-Specific Computation dengan topik **Sekuens DNA & RNA**.

Model utama pada proyek ini adalah **binary promoter classification**:

- Input: DNA sequence string
- Output:
  - `0 = non-promoter`
  - `1 = promoter`

Aplikasi juga menyediakan utilitas biologis untuk sequence analysis, central dogma, ORF detection, mutation explainer, dan Needleman-Wunsch alignment.

## Dataset

Primary dataset:

- `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`

Selected task:

- `promoter_all`

Dataset ini digunakan melalui official `train` dan `test` splits, lalu difilter dengan:

```text
task == "promoter_all"
```

Synthetic, randomly generated, toy, placeholder, or manually invented sequences are **not used** for model training, model evaluation, default demo examples, or reported results. Unit tests may use tiny toy strings only for deterministic biological utility functions such as validation, transcription, translation, GC content, k-mer extraction, alignment, and mutation classification.

If the real dataset cannot be downloaded or validated, the pipeline fails loudly. It does not silently create fake data.

## Methods

- Sequence validation: DNA input must contain only `A`, `C`, `G`, `T`, `N`.
- Feature extraction: overlapping k-mers.
- Baseline model: Logistic Regression by default.
- Optional model choices: Linear SVM and Random Forest.
- Evaluation: official real `promoter_all` test split only.
- Biological analysis: GC content, motifs, ORF, transcription, translation, mutation impact, and global alignment.

## Tech Stack

- Python 3.10+
- Streamlit
- NumPy, pandas
- scikit-learn
- Hugging Face `datasets`
- joblib
- matplotlib
- pytest

## Reproducible Workflow

The commands below are written as single-line commands so they work cleanly in PowerShell, Command Prompt, Git Bash, and most terminals.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download real dataset

```bash
python scripts/download_dataset.py --dataset InstaDeepAI/nucleotide_transformer_downstream_tasks_revised --task promoter_all --output-dir data/processed
```

This creates:

- `data/processed/promoter_all_train.csv`
- `data/processed/promoter_all_test.csv`

Each CSV contains exactly:

- `sequence`
- `label`
- `task`
- `name`

### 3. Train baseline model

```bash
python scripts/train_baseline.py --train-csv data/processed/promoter_all_train.csv --model-output models/promoter_kmer_logreg.joblib --vectorizer-output models/promoter_kmer_vectorizer.joblib --k 6 --model logistic_regression
```

The script also writes:

```text
models/promoter_kmer_metadata.json
```

### 4. Evaluate model

```bash
python scripts/evaluate_model.py --test-csv data/processed/promoter_all_test.csv --model-path models/promoter_kmer_logreg.joblib --vectorizer-path models/promoter_kmer_vectorizer.joblib --output-dir reports/evaluation
```

Evaluation artifacts:

- `reports/evaluation/metrics.json`
- `reports/evaluation/classification_report.txt`
- `reports/evaluation/confusion_matrix.png`

Metrics explicitly reference:

- dataset: `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`
- task: `promoter_all`

### 5. Run app

```bash
streamlit run app.py
```

The app loads:

- `models/promoter_kmer_logreg.joblib`
- `models/promoter_kmer_vectorizer.joblib`

Example sequences in the app must come from:

```text
data/processed/promoter_all_test.csv
```

Use the **Load random real test sequence** button after downloading the dataset.

## Notebook Pipeline

Untuk melihat pipeline proyek secara end-to-end dalam satu tempat, gunakan notebook:

```text
notebooks/promoter_all_pipeline.ipynb
```

Notebook tersebut menuliskan kode Python secara eksplisit untuk:

- load dataset real dari Hugging Face,
- filter `task == "promoter_all"`,
- validasi dan simpan CSV,
- membangun k-mer `CountVectorizer`,
- melatih `LogisticRegression`,
- menghitung metrik evaluasi,
- menyimpan model, vectorizer, metadata, dan report,
- memprediksi satu sekuens dari real test split.

Script di `scripts/` tetap tersedia sebagai alternatif otomatisasi untuk menjalankan langkah yang sama dari terminal.

## Quality Gates

```bash
python -m compileall src app.py scripts
pytest -q
python scripts/download_dataset.py --dataset InstaDeepAI/nucleotide_transformer_downstream_tasks_revised --task promoter_all --output-dir data/processed
python scripts/train_baseline.py --train-csv data/processed/promoter_all_train.csv --model-output models/promoter_kmer_logreg.joblib --vectorizer-output models/promoter_kmer_vectorizer.joblib --k 6 --model logistic_regression
python scripts/evaluate_model.py --test-csv data/processed/promoter_all_test.csv --model-path models/promoter_kmer_logreg.joblib --vectorizer-path models/promoter_kmer_vectorizer.joblib --output-dir reports/evaluation
streamlit run app.py
```

## Limitations

- This is a baseline k-mer model for academic demonstration.
- Predictions are not scientifically or clinically definitive.
- Scientific conclusions require careful dataset understanding, validation protocol, and domain review.
- The current implementation focuses on the k-mer Logistic Regression baseline and explainable biological analysis modules.
