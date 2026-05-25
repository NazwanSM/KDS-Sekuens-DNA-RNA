from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dna_rna_classifier.central_dogma import find_orfs, transcribe, translate_dna, translate_rna  # noqa: E402
from dna_rna_classifier.features import find_motifs, sequence_feature_summary  # noqa: E402
from dna_rna_classifier.mutation import analyze_mutations  # noqa: E402
from dna_rna_classifier.mutation_sensitivity import (  # noqa: E402
    format_sensitivity_interpretation,
    scan_mutation_sensitivity,
)
from dna_rna_classifier.plotting import composition_dataframe, kmer_dataframe  # noqa: E402
from dna_rna_classifier.promoter_dataset import (  # noqa: E402
    DATASET_NAME,
    LABEL_MAPPING,
    TASK_NAME,
    read_promoter_csv,
)
from dna_rna_classifier.validation import detect_sequence_type, parse_fasta_text, validate_sequence  # noqa: E402
from dna_rna_classifier.visualization_text import (  # noqa: E402
    summarize_mutation_analysis,
    summarize_sequence_features,
    truncate_sequence,
)

DEFAULT_MODEL_PATH = Path("models/promoter_kmer_logreg.joblib")
DEFAULT_VECTORIZER_PATH = Path("models/promoter_kmer_vectorizer.joblib")
DEFAULT_TEST_CSV = Path("data/processed/promoter_all_test.csv")
DEFAULT_REPORT_DIR = Path("reports/evaluation")

def _read_uploaded_file(uploaded_file: object | None) -> str:
    """Decode an uploaded Streamlit file into text."""
    if uploaded_file is None:
        return ""
    return uploaded_file.getvalue().decode("utf-8")

def _first_sequence_from_input(text: str) -> tuple[str, str]:
    """Parse text/FASTA input and return the first record id and sequence."""
    records = parse_fasta_text(text)
    if not records:
        raise ValueError("Please enter a DNA/RNA sequence or upload a FASTA file.")
    return records[0]

def _sequence_translation(sequence: str, frame: int) -> tuple[str, str]:
    """Return transcription text and protein translation for display."""
    sequence_type = detect_sequence_type(sequence)
    if sequence_type == "RNA":
        return "Input is RNA; transcription is not applied.", translate_rna(sequence, frame=frame)
    rna = transcribe(sequence)
    return rna, translate_dna(sequence, frame=frame)

def _validate_dna_for_promoter_model(sequence: str) -> str:
    """Validate model input as DNA only."""
    cleaned = sequence.strip().upper()
    if not cleaned:
        raise ValueError("Promoter classification requires a non-empty DNA sequence.")
    if not set(cleaned).issubset(set("ACGTN")):
        raise ValueError("Promoter classification accepts DNA bases only: A, C, G, T, N.")
    return cleaned

def _download_command() -> str:
    """Return the canonical dataset download command."""
    return (
        "python scripts/download_dataset.py "
        "--dataset InstaDeepAI/nucleotide_transformer_downstream_tasks_revised "
        "--task promoter_all "
        "--output-dir data/processed"
    )

def _train_command() -> str:
    """Return the canonical training command."""
    return (
        "python scripts/train_baseline.py "
        "--train-csv data/processed/promoter_all_train.csv "
        "--model-output models/promoter_kmer_logreg.joblib "
        "--vectorizer-output models/promoter_kmer_vectorizer.joblib "
        "--k 6 "
        "--model logistic_regression"
    )

def _mutation_sensitivity_command() -> str:
    """Return the canonical mutation sensitivity command."""
    return (
        "python scripts/analyze_mutation_sensitivity.py "
        "--test-csv data/processed/promoter_all_test.csv "
        "--only-promoters "
        "--sample-size 5 "
        "--model-path models/promoter_kmer_logreg.joblib "
        "--vectorizer-path models/promoter_kmer_vectorizer.joblib "
        "--k 6 "
        "--output-dir reports/mutation_sensitivity"
    )

def _changed_kmers_label(changed_kmers: dict) -> str:
    """Compact changed k-mer pairs for table display."""
    pairs = changed_kmers.get("changed_pairs", [])
    return "; ".join(
        f"{pair['start']}-{pair['end']}: {pair['original_kmer']}->{pair['mutated_kmer']}"
        for pair in pairs
    )

def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="DNA/RNA Promoter Classifier", layout="wide")
    st.title("DNA/RNA Sequence Function Classifier + Mutation Explainer")
    st.info("Model trained on real promoter_all data from InstaDeepAI/nucleotide_transformer_downstream_tasks_revised.")
    st.warning(
        "Predictions are baseline computational estimates for academic demonstration; "
        "they are not scientifically or clinically definitive."
    )

    with st.sidebar:
        st.header("Configuration")
        k = st.slider("k-mer size for analysis", min_value=2, max_value=6, value=6)
        model_path = Path(st.text_input("Model path", value=str(DEFAULT_MODEL_PATH)))
        vectorizer_path = Path(st.text_input("Vectorizer path", value=str(DEFAULT_VECTORIZER_PATH)))
        test_csv_path = Path(st.text_input("Real test CSV", value=str(DEFAULT_TEST_CSV)))
        reading_frame = st.selectbox("Reading frame", options=[0, 1, 2], index=0)
        orf_min_length = st.slider("ORF minimum length", min_value=0, max_value=300, value=30, step=3)
        st.subheader("Alignment scoring")
        match_score = st.number_input("Match", value=2, step=1)
        mismatch_score = st.number_input("Mismatch", value=-1, step=1)
        gap_score = st.number_input("Gap", value=-2, step=1)

    tab_analysis, tab_classification, tab_sensitivity, tab_mutation, tab_evaluation, tab_about = st.tabs(
        [
            "Sequence Analysis",
            "Classification",
            "Promoter Mutation Sensitivity",
            "Mutation Explainer",
            "Model Evaluation",
            "About",
        ]
    )

    with tab_analysis:
        st.subheader("Sequence Analysis")
        if test_csv_path.exists() and st.button("Load random real test sequence", key="analysis_random"):
            try:
                test_df = read_promoter_csv(test_csv_path, task=TASK_NAME)
                sample = test_df.sample(1).iloc[0]
                st.session_state["last_sequence"] = sample["sequence"]
                st.session_state["last_sequence_name"] = sample["name"]
                st.session_state["last_sequence_label"] = int(sample["label"])
            except Exception as exc:
                st.warning(str(exc))
        elif not test_csv_path.exists():
            st.info("Real test CSV is missing. Run this command before loading examples:")
            st.code(_download_command(), language="bash")

        sequence_text = st.text_area(
            "DNA/RNA input",
            value=st.session_state.get("last_sequence", ""),
            height=150,
            placeholder="Paste a DNA/RNA sequence or load a real test sequence.",
        )
        uploaded = st.file_uploader("FASTA upload", type=["fa", "fasta", "txt"])
        if st.button("Analyze", type="primary"):
            try:
                input_text = _read_uploaded_file(uploaded) or sequence_text
                record_id, sequence = _first_sequence_from_input(input_text)
                st.session_state["last_sequence"] = sequence

                summary = sequence_feature_summary(sequence, k=k)
                st.success(f"Analyzed record: {record_id}")
                if "last_sequence_name" in st.session_state:
                    st.caption(
                        f"Loaded real test row: {st.session_state['last_sequence_name']} "
                        f"(label {st.session_state.get('last_sequence_label')})"
                    )
                st.write(summarize_sequence_features(summary))

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Type", summary["sequence_type"])
                col2.metric("Length", summary["length"])
                col3.metric("GC content", f"{summary['gc_content']}%")
                col4.metric("AT/AU content", f"{summary['at_or_au_content']}%")

                st.write("Base composition")
                comp_df = composition_dataframe(sequence)
                st.dataframe(comp_df, use_container_width=True)
                st.bar_chart(comp_df.set_index("base")["count"])

                st.write("Top k-mers")
                st.dataframe(kmer_dataframe(sequence, k=k, top_n=10), use_container_width=True)

                motifs = ["AUG", "UAA", "UAG", "UGA"] if detect_sequence_type(sequence) == "RNA" else [
                    "TATA",
                    "ATG",
                    "TAA",
                    "TAG",
                    "TGA",
                ]
                st.write("Motif hits")
                st.dataframe(pd.DataFrame(find_motifs(sequence, motifs)), use_container_width=True)

                st.write("Open reading frames")
                orfs = find_orfs(sequence, min_length=orf_min_length)
                if orfs:
                    orf_df = pd.DataFrame(orfs)
                    orf_df["sequence"] = orf_df["sequence"].map(lambda value: truncate_sequence(value, 80))
                    st.dataframe(orf_df, use_container_width=True)
                else:
                    st.info("No ORFs found with the selected minimum length.")

                transcription_result, protein = _sequence_translation(sequence, frame=reading_frame)
                st.write("Transcription")
                st.code(transcription_result)
                st.write("Translation")
                st.code(protein or "(no complete codons translated)")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:  # pragma: no cover - UI guard
                st.error(f"Unexpected analysis error: {exc}")

    with tab_classification:
        st.subheader("Promoter Classification")
        st.write(f"Dataset: `{DATASET_NAME}`")
        st.write(f"Task: `{TASK_NAME}` with labels `0 = non-promoter`, `1 = promoter`.")

        if test_csv_path.exists() and st.button("Load random real test sequence", key="classification_random"):
            try:
                test_df = read_promoter_csv(test_csv_path, task=TASK_NAME)
                sample = test_df.sample(1).iloc[0]
                st.session_state["last_sequence"] = sample["sequence"]
                st.session_state["last_sequence_name"] = sample["name"]
                st.session_state["last_sequence_label"] = int(sample["label"])
            except Exception as exc:
                st.warning(str(exc))
        elif not test_csv_path.exists():
            st.info("Real test CSV is missing. Run this command before loading examples:")
            st.code(_download_command(), language="bash")

        classify_sequence = st.text_area(
            "DNA sequence to classify",
            value=st.session_state.get("last_sequence", ""),
            height=140,
            placeholder="Paste a DNA sequence or load a real test sequence.",
        )
        if "last_sequence_name" in st.session_state:
            st.caption(
                f"Current real test row: {st.session_state['last_sequence_name']} "
                f"(true label {st.session_state.get('last_sequence_label')})"
            )

        if st.button("Predict promoter label"):
            try:
                if not model_path.exists() or not vectorizer_path.exists():
                    st.warning("Model/vectorizer files are not available yet.")
                    st.code(_train_command(), language="bash")
                else:
                    cleaned = _validate_dna_for_promoter_model(classify_sequence)
                    model = joblib.load(model_path)
                    vectorizer = joblib.load(vectorizer_path)
                    features = vectorizer.transform([cleaned])
                    prediction = int(model.predict(features)[0])
                    st.metric("Predicted label", f"{prediction} ({LABEL_MAPPING.get(prediction, 'unknown')})")
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)[0]
                        classes = [int(value) for value in model.classes_]
                        prob_df = pd.DataFrame(
                            [
                                {
                                    "class": label,
                                    "meaning": LABEL_MAPPING.get(label, "unknown"),
                                    "probability": float(probability),
                                }
                                for label, probability in zip(classes, probabilities)
                            ]
                        )
                        st.dataframe(prob_df, use_container_width=True)
                        st.bar_chart(prob_df.set_index("meaning")["probability"])
                    elif hasattr(model, "decision_function"):
                        st.write("Classifier does not provide calibrated probabilities.")
                        st.json({"decision_score": float(model.decision_function(features)[0])})
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:  # pragma: no cover - UI guard
                st.error(f"Unexpected prediction error: {exc}")

    with tab_sensitivity:
        st.subheader("Promoter Mutation Sensitivity Analyzer")
        st.write(
            "This analysis uses in-silico point mutations to measure how much each mutation "
            "changes the trained model's promoter probability."
        )
        st.warning(
            "Sensitive positions are model-sensitive regions, not experimentally validated biological motifs."
        )
        top_n = st.slider("Top disruptive mutations", min_value=5, max_value=30, value=10)
        show_changed_kmers = st.checkbox("Show changed k-mers", value=False)

        sensitivity_sequence = st.text_area(
            "DNA sequence for sensitivity scan",
            value=st.session_state.get("last_sequence", ""),
            height=140,
            placeholder="Paste a DNA sequence or load a real test sequence from another tab.",
        )

        if st.button("Run mutation sensitivity scan"):
            try:
                if not model_path.exists() or not vectorizer_path.exists():
                    st.warning("Model/vectorizer files are not available yet.")
                    st.code(_train_command(), language="bash")
                else:
                    cleaned = _validate_dna_for_promoter_model(sensitivity_sequence)
                    model = joblib.load(model_path)
                    vectorizer = joblib.load(vectorizer_path)
                    scan_result = scan_mutation_sensitivity(cleaned, model, vectorizer, k=k)
                    original = scan_result["original_prediction"]
                    robustness = scan_result["robustness_summary"]

                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Original predicted label",
                        f"{original['predicted_label']} ({LABEL_MAPPING.get(original['predicted_label'], 'unknown')})",
                    )
                    if original.get("promoter_probability") is not None:
                        col2.metric("Original promoter probability", f"{original['promoter_probability']:.4f}")
                    else:
                        col2.metric("Original promoter score", f"{original['promoter_score']:.4f}")

                    st.write("Robustness summary")
                    summary_df = pd.DataFrame(
                        [
                            {
                                key: value
                                for key, value in robustness.items()
                                if key != "score_type"
                            }
                        ]
                    )
                    st.dataframe(summary_df, use_container_width=True)

                    top_rows = scan_result["top_disruptive_mutations"][:top_n]
                    top_df = pd.DataFrame(
                        [
                            {
                                "mutation_label": row["mutation_label"],
                                "position_1based": row["position_1based"],
                                "original_base": row["original_base"],
                                "mutant_base": row["mutant_base"],
                                "original_probability": row["original_probability"],
                                "mutant_probability": row["mutant_probability"],
                                "probability_drop": row["probability_drop"],
                                "delta_score": row["delta_score"],
                                "changed_kmers": _changed_kmers_label(row["changed_kmers"]),
                            }
                            for row in top_rows
                        ]
                    )
                    if not show_changed_kmers and "changed_kmers" in top_df:
                        top_df = top_df.drop(columns=["changed_kmers"])
                    st.write("Top disruptive mutations")
                    st.dataframe(top_df, use_container_width=True)

                    position_df = pd.DataFrame(scan_result["position_sensitivity"])
                    if not position_df.empty:
                        chart_metric = (
                            "max_probability_drop"
                            if position_df["max_probability_drop"].notna().any()
                            else "mean_abs_delta_score"
                        )
                        st.write("Position sensitivity chart")
                        st.line_chart(position_df.set_index("position_1based")[chart_metric])

                    st.write("Interpretation")
                    st.info(format_sensitivity_interpretation(scan_result))
                    st.caption("CLI report command:")
                    st.code(_mutation_sensitivity_command(), language="bash")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Unexpected mutation sensitivity error: {exc}")

    with tab_mutation:
        st.subheader("Mutation Explainer")
        ref_sequence = st.text_area(
            "Reference / normal sequence",
            value="",
            height=120,
            placeholder="Paste a reference DNA/RNA sequence.",
        )
        mutated_sequence = st.text_area(
            "Mutated sequence",
            value="",
            height=120,
            placeholder="Paste the mutated DNA/RNA sequence.",
        )
        if st.button("Analyze mutation"):
            try:
                result = analyze_mutations(ref_sequence, mutated_sequence, frame=reading_frame)
                st.write(summarize_mutation_analysis(result))

                if result["mutations"]:
                    st.write("Mutation table")
                    st.dataframe(pd.DataFrame(result["mutations"]), use_container_width=True)
                else:
                    st.info("No sequence-level mutations detected.")

                if result["codon_effects"]:
                    st.write("Codon-level effects")
                    st.dataframe(pd.DataFrame(result["codon_effects"]), use_container_width=True)

                st.write("Protein comparison")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"sequence": "reference", "protein": result["reference_protein"]},
                            {"sequence": "mutated", "protein": result["mutated_protein"]},
                        ]
                    ),
                    use_container_width=True,
                )

                display_alignment = result["alignment"]
                if (match_score, mismatch_score, gap_score) != (2, -1, -2):
                    from dna_rna_classifier.alignment import format_alignment, needleman_wunsch

                    clean_ref = validate_sequence(ref_sequence)
                    clean_mut = validate_sequence(mutated_sequence)
                    custom = needleman_wunsch(
                        clean_ref,
                        clean_mut,
                        match=int(match_score),
                        mismatch=int(mismatch_score),
                        gap=int(gap_score),
                    )
                    display_alignment = {
                        **custom,
                        "formatted": format_alignment(custom["aligned_seq1"], custom["aligned_seq2"]),
                    }
                    st.caption("Alignment below uses the custom sidebar scoring.")
                st.metric("Needleman-Wunsch score", display_alignment["score"])
                st.metric("Identity percentage", f"{display_alignment['metrics']['identity_percentage']}%")
                st.code(display_alignment["formatted"])
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:  # pragma: no cover - UI guard
                st.error(f"Unexpected mutation analysis error: {exc}")

    with tab_evaluation:
        st.subheader("Model Evaluation")
        metrics_path = DEFAULT_REPORT_DIR / "metrics.json"
        report_path = DEFAULT_REPORT_DIR / "classification_report.txt"
        cm_path = DEFAULT_REPORT_DIR / "confusion_matrix.png"
        if not metrics_path.exists():
            st.warning("Evaluation artifacts are missing.")
            st.code(
                "python scripts/evaluate_model.py "
                "--test-csv data/processed/promoter_all_test.csv "
                "--model-path models/promoter_kmer_logreg.joblib "
                "--vectorizer-path models/promoter_kmer_vectorizer.joblib "
                "--output-dir reports/evaluation",
                language="bash",
            )
        else:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.json(metrics)
            if report_path.exists():
                st.text(report_path.read_text(encoding="utf-8"))
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion matrix on real promoter_all test data")

    with tab_about:
        st.subheader("About")
        st.markdown(
            """
            **Model objective:** binary promoter classification from DNA sequence strings.

            **Dataset:** `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`.

            **Task:** `promoter_all`, where `0 = non-promoter` and `1 = promoter`.

            **Synthetic data policy:** synthetic, random, placeholder, or manually invented sequences are not used for model training, evaluation, default examples, or reported results.

            **k-mers:** overlapping subsequences of length k used as baseline machine learning features.

            **Central dogma:** DNA can be transcribed to RNA and RNA translated to amino-acid sequences.

            **Mutation types:** silent mutations preserve amino acids, missense mutations change amino acids, nonsense mutations introduce stop codons, and frameshifts alter the reading frame.

            **Sequence alignment:** Needleman-Wunsch global alignment compares two sequences across their full length.

            **Promoter mutation sensitivity:** in-silico point mutations are used only as a post-hoc interpretation layer over the trained k-mer classifier. Sensitive positions are model-sensitive regions, not experimentally validated promoter motifs.
            """
        )

if __name__ == "__main__":
    main()
