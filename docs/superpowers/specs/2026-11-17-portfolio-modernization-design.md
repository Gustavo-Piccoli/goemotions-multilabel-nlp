# GoEmotions Multi-Label Portfolio Modernization — Design Spec

**Date:** 2026-11-17
**Author:** Gustavo Auad Piccoli
**Source Repository:** `https://github.com/Gustavo-Piccoli-Academico/Trabalho_Final_PLN` (preserved untouched)
**Target Repository:** `goemotions-multilabel-nlp` (new repository)

---

## 1. Purpose & Goals

Transform the academic project `Trabalho_Final_PLN` (PPGC/UFRGS — Fine-Grained Multi-Label Emotion Classification on GoEmotions with BERT and RoBERTa) into a production-grade Machine Learning & NLP portfolio repository while preserving the original repository untouched.

Key goals:
1. **Repository Structure:** Promote code from `Implementação/` to the repository root following modern Python packaging standards (`src/`, `configs/`, `notebooks/`, `results/`, `docs/`, `tests/`, `.github/workflows/`).
2. **Artifact Hygiene:** Remove heavy/intermediate training binaries (`checkpoint-2037/`, `best_model/` tokenizer/training_args blobs) from Git tracking while preserving all analytical outputs (`metrics.json`, `classification_report.csv`, `per_class_f1.csv`, `confusion_gold_pred.csv/.png`, `sentiment_confusion.csv/.png`, `error_examples.csv`) and adding a visual comparison chart (`results/comparison_bert_roberta.png`). Move academic PDFs to `docs/article.pdf` and `docs/presentation.pdf`.
3. **ML Engineering Enhancements:**
   - Add `pyproject.toml` with package metadata, dependencies, CLI entrypoints, and `ruff` / `pytest` configuration.
   - Extract testable multi-hot encoding helper in `src/dataset.py` (`encode_multihot`).
   - Add `src/predict.py` providing both a programmatic API (`predict_texts`, `format_predictions`) and a CLI (`python -m src.predict`) for multi-label emotion inference with sentiment group mapping and top-k probability display.
   - Add `src/compare.py` (or comparative plotting function in `src/evaluate.py`) to generate `results/comparison_bert_roberta.png` directly from `results/bert` and `results/roberta` artifacts.
4. **Testing & CI:**
   - Add unit tests in `tests/` (`test_labels.py`, `test_metrics.py`, `test_dataset.py`, `test_predict.py`, `test_compare.py`) that run fast on CPU without downloading multi-gigabyte weights.
   - Add `.github/workflows/ci.yml` running `ruff` and `pytest` on push/PR.
5. **Portfolio-Grade Bilingual Documentation (`README.md`):**
   - Portuguese main body with an **English Executive Summary** at the top.
   - Badges (Python, PyTorch, Hugging Face Transformers, License, CI).
   - Research question, methodology, controlled hyperparameter table, quantitative results (aggregate + per-class highlights + sentiment group confusion matrices), qualitative error analysis (missed detections vs. semantic neighbors), embedded charts, reproduction guide, and citation.

---

## 2. Target Architecture & Directory Layout

```text
goemotions-multilabel-nlp/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: Ruff lint + Pytest on CPU
├── configs/
│   ├── bert.yaml                   # Baseline configuration (bert-base-uncased)
│   └── roberta.yaml                # Main model configuration (roberta-base)
├── docs/
│   ├── article.pdf                 # Full academic paper (SBC format, UFRGS)
│   ├── presentation.pdf            # Slide deck (PPGC/UFRGS)
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-11-17-portfolio-modernization-design.md
│       └── plans/
│           └── 2026-11-17-portfolio-modernization.md
├── notebooks/
│   └── colab_train.ipynb           # Updated Google Colab notebook pointing to root paths
├── results/
│   ├── comparison_bert_roberta.png # Visual comparison (global metrics + top class gains)
│   ├── bert/
│   │   ├── classification_report.csv
│   │   ├── confusion_gold_pred.csv
│   │   ├── confusion_gold_pred.png
│   │   ├── error_examples.csv
│   │   ├── metrics.json
│   │   ├── per_class_f1.csv
│   │   ├── sentiment_confusion.csv
│   │   └── sentiment_confusion.png
│   └── roberta/
│       ├── classification_report.csv
│       ├── confusion_gold_pred.csv
│       ├── confusion_gold_pred.png
│       ├── error_examples.csv
│       ├── metrics.json
│       ├── per_class_f1.csv
│       ├── sentiment_confusion.csv
│       └── sentiment_confusion.png
├── src/
│   ├── __init__.py
│   ├── labels.py                   # 28-class taxonomy + 4 sentiment groups
│   ├── dataset.py                  # GoEmotions loader, encode_multihot, MultiLabelCollator
│   ├── metrics.py                  # Sigmoid, logits_to_preds, Macro/Micro/Weighted F1, Subset Acc
│   ├── train.py                    # HuggingFace Trainer fine-tuning pipeline
│   ├── evaluate.py                 # Detailed evaluation, confusion matrices, error sampling
│   ├── compare.py                  # Generates comparison_bert_roberta.png from results/
│   └── predict.py                  # Inference API & CLI for custom texts
├── tests/
│   ├── test_labels.py
│   ├── test_metrics.py
│   ├── test_dataset.py
│   ├── test_predict.py
│   └── test_compare.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Component Specifications

### 3.1 `src/dataset.py`
- Extract `encode_multihot(label_Lists: list[list[int]], num_labels: int = NUM_LABELS) -> list[list[float]]` so multi-hot encoding can be unit-tested independently of HuggingFace dataset downloads.
- Keep `load_goemotions` and `MultiLabelCollator` with clean type annotations and docstrings.

### 3.2 `src/predict.py`
- `predict_emotions(texts: list[str], model, tokenizer, threshold: float = 0.5, top_k: int = 5, max_length: int = 64, device: str | None = None) -> list[dict[str, Any]]`:
  - Tokenizes `texts`, runs forward pass with `torch.no_grad()`, applies `sigmoid` from `src.metrics`.
  - Returns for each text:
    - `text`: input string
    - `active_emotions`: list of dicts `{"emotion": label, "probability": float, "sentiment": group}` for all classes with `prob >= threshold`, sorted descending by probability.
    - `top_k`: list of top `k` dicts `{"emotion": label, "probability": float, "sentiment": group}` regardless of threshold.
    - `dominant_sentiment`: set/list of activated sentiment groups (or `"neutral (below threshold)"` if no label exceeds threshold).
- `format_prediction_report(results: list[dict[str, Any]], threshold: float = 0.5) -> str`:
  - Produces a clean, human-readable CLI report.
- `main()`:
  - CLI accepting `--model_dir` (local path or HuggingFace Hub model ID), `--text` (one or more strings), `--threshold` (default `0.5`), `--top_k` (default `5`), `--json` (optional JSON output).
  - Checks if `--model_dir` looks like an untrained local path (e.g. `results/roberta/best_model` when missing on disk) and raises a friendly, actionable error message instructing the user how to train or pass a Hub model ID.

### 3.3 `src/compare.py`
- Loads `results/bert/metrics.json`, `results/roberta/metrics.json`, `results/bert/per_class_f1.csv`, and `results/roberta/per_class_f1.csv`.
- Generates a high-resolution two-panel figure `results/comparison_bert_roberta.png`:
  - **Left panel:** Grouped bar chart comparing BERT vs. RoBERTa across the 4 global test metrics (`Macro F1`, `Micro F1`, `Weighted F1`, `Subset Accuracy`) with percentage annotations.
  - **Right panel:** Horizontal bar chart comparing per-class F1 scores on the top 15 classes sorted by RoBERTa F1, visually highlighting the 11 mid-frequency emotions unlocked by RoBERTa where BERT scored 0.0.

### 3.4 `README.md`
- **Header:** Title, badges (CI, Python 3.10+, PyTorch, Hugging Face, MIT License).
- **Executive Summary (English):** Concise overview of the problem, methodology, and key findings (`RoBERTa-base` nearly doubles `BERT-base` Macro F1 from `13.7%` to `26.4%` and expands active class coverage from `6/28` to `17/28` under identical hyperparameters).
- **Visão Geral & Motivação (PT-BR):** Why fine-grained multi-label emotion classification matters beyond polarity.
- **Principais Resultados:**
  - Embedded `results/comparison_bert_roberta.png`.
  - Aggregate metrics table (Validation & Test).
  - Per-class breakdown table showing Head classes (`gratitude`, `amusement`, `love`, `admiration`, `neutral`) vs. Mid-frequency classes unlocked only by RoBERTa (`joy`, `optimism`, `sadness`, `surprise`, `approval`, `curiosity`, `anger`, `disapproval`, `confusion`) vs. Tail classes (`grief`, `relief`, `pride`).
  - Side-by-side comparison of Sentiment Group Confusion Matrices (`results/bert/sentiment_confusion.png` and `results/roberta/sentiment_confusion.png`) and 28×28 matrices.
  - Qualitative Error Analysis table with real examples from the test set (Missed detections due to fixed 0.5 threshold vs. Semantic neighbor confusions like `pride -> admiration` and `relief -> joy`).
- **Arquitetura do Projeto & Engenharia:** Directory tree and modular pipeline diagram (Mermaid).
- **Como Executar (Quickstart):** Installation (`pip install -e ".[dev]"`), smoke test, full training, evaluation, comparative plotting, inference CLI (`python -m src.predict`), and unit tests (`pytest`).
- **Documentação Acadêmica & Citação:** Links to `docs/article.pdf` and `docs/presentation.pdf` + BibTeX entry.
