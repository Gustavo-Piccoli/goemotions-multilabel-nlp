# GoEmotions Multi-Label Portfolio Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the `Trabalho_Final_PLN` academic project into a clean, well-tested, portfolio-ready ML/NLP repository (`goemotions-multilabel-nlp`) while keeping the original repository untouched.

**Architecture:** Promote the codebase from `Implementação/` to the repository root, package it with `pyproject.toml`, remove intermediate training binaries from Git while keeping analytical artifacts (`CSV`, `JSON`, `PNG`), add an inference CLI (`src/predict.py`) and comparative visualization generator (`src/compare.py`), add fast CPU unit tests (`tests/`) + GitHub Actions CI (`.github/workflows/ci.yml`), and write a comprehensive bilingual (PT-BR + EN Executive Summary) `README.md`.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face (`transformers`, `datasets`, `accelerate`), `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `PyYAML`, `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-11-17-portfolio-modernization-design.md`

## Global Constraints

- Do NOT modify or delete the remote repository `https://github.com/Gustavo-Piccoli-Academico/Trabalho_Final_PLN`. All work happens in the new repository `C:\Users\gp116459\Desktop\Repositórios\goemotions-multilabel-nlp`.
- Remove heavy/intermediate training checkpoints (`checkpoint-2037/`, `best_model/` binary/tokenizer blobs) from Git tracking; retain all analytical CSVs, JSONs, and PNGs in `results/bert/` and `results/roberta/`.
- Move `TF PLN Gustavo.pdf` to `docs/article.pdf` and `Apresentação_E_Repositório.pdf` to `docs/presentation.pdf`.
- All unit tests in `tests/` must run in under 15 seconds on CPU without downloading multi-hundred-megabyte weights from Hugging Face Hub.

---

### Task 1: Repository Restructuring, Artifact Cleanup & Packaging

**Files:**
- Move: `Implementação/src/*` -> `src/*`
- Move: `Implementação/configs/*` -> `configs/*`
- Move: `Implementação/notebooks/colab_train.ipynb` -> `notebooks/colab_train.ipynb`
- Move: `Implementação/results/bert/*.{csv,json,png}` -> `results/bert/`
- Move: `Implementação/results/roberta/*.{csv,json,png}` -> `results/roberta/`
- Move: `TF PLN Gustavo.pdf` -> `docs/article.pdf`
- Move: `Apresentação_E_Repositório.pdf` -> `docs/presentation.pdf`
- Delete: `Implementação/` (including `checkpoint-2037` and `best_model` folders)
- Create: `.gitignore`, `LICENSE`, `pyproject.toml`, `requirements.txt`

**Interfaces:**
- Produces: Clean root layout with `src`, `configs`, `results`, `docs`, `notebooks`, `pyproject.toml`, `requirements.txt`.

- [ ] **Step 1: Move files to root structure and delete `Implementação/`**
- [ ] **Step 2: Create `.gitignore`, `LICENSE` (MIT), `requirements.txt`, and `pyproject.toml`**
- [ ] **Step 3: Verify directory layout**

---

### Task 2: Core Modules Refinement & Unit Tests (`labels`, `metrics`, `dataset`)

**Files:**
- Modify: `src/__init__.py`
- Modify: `src/dataset.py`
- Create: `tests/test_labels.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `src.labels`, `src.metrics`
- Produces: `encode_multihot(label_lists: list[list[int]], num_labels: int = NUM_LABELS) -> list[list[float]]` in `src/dataset.py`

- [ ] **Step 1: Write unit tests in `tests/test_labels.py`, `tests/test_metrics.py`, and `tests/test_dataset.py`**
- [ ] **Step 2: Run `pytest` to verify `test_dataset.py` fails before `encode_multihot` is added**
- [ ] **Step 3: Implement `encode_multihot` in `src/dataset.py` and update `src/__init__.py`**
- [ ] **Step 4: Run `pytest` to verify all core tests pass**

---

### Task 3: Multi-Label Inference CLI & Unit Tests (`src/predict.py`)

**Files:**
- Create: `src/predict.py`
- Create: `tests/test_predict.py`

**Interfaces:**
- Consumes: `src.labels.{GOEMOTIONS_LABELS, LABEL2SENTIMENT}`, `src.metrics.sigmoid`
- Produces: `predict_emotions(...)`, `format_prediction_report(...)`, and CLI `python -m src.predict`

- [ ] **Step 1: Write unit tests in `tests/test_predict.py` using lightweight mock model & tokenizer**
- [ ] **Step 2: Run `pytest tests/test_predict.py` to verify failure**
- [ ] **Step 3: Implement `src/predict.py` with `predict_emotions`, `format_prediction_report`, and CLI `main()`**
- [ ] **Step 4: Run `pytest tests/test_predict.py` to verify all tests pass**

---

### Task 4: Comparative Visualization Generator (`src/compare.py`) & Chart Asset

**Files:**
- Create: `src/compare.py`
- Create: `tests/test_compare.py`
- Create: `results/comparison_bert_roberta.png`

**Interfaces:**
- Consumes: `results/bert/metrics.json`, `results/roberta/metrics.json`, `results/bert/per_class_f1.csv`, `results/roberta/per_class_f1.csv`
- Produces: `generate_comparison_chart(results_dir: Path, output_path: Path) -> Path` and `results/comparison_bert_roberta.png`

- [ ] **Step 1: Write unit test in `tests/test_compare.py`**
- [ ] **Step 2: Implement `src/compare.py` and generate `results/comparison_bert_roberta.png`**
- [ ] **Step 3: Run `pytest tests/test_compare.py` and verify `results/comparison_bert_roberta.png` exists**

---

### Task 5: CI Workflow, Notebook Update & Portfolio `README.md`

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `notebooks/colab_train.ipynb`
- Create: `README.md`

**Interfaces:**
- Consumes: All modules, test suite, and analytical results in `results/` and `docs/`
- Produces: Complete portfolio documentation and CI pipeline

- [ ] **Step 1: Create `.github/workflows/ci.yml` for GitHub Actions (Ruff + Pytest)**
- [ ] **Step 2: Update `notebooks/colab_train.ipynb` to remove hardcoded private token references and support public/portfolio usage**
- [ ] **Step 3: Write comprehensive `README.md` (EN Executive Summary + PT-BR deep dive, tables, charts, CLI usage, architecture, and citation)**
- [ ] **Step 4: Run full test suite (`pytest -v`) and linter (`ruff check .`), then commit all changes to the new repository**
