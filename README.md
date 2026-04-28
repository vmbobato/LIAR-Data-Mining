# Truth-Risk Intelligence: Detecting Low-Truth Political Claims with Text and Context

**Final curated notebook:** [`main_notebook.ipynb`](main_notebook.ipynb)

## Overview
This repository contains a data mining project on the LIAR dataset (PolitiFact fact-checked political statements). The project investigates whether combining text representations with speaker/topic context improves detection of low-truth claims, while reducing speaker memorization bias. The repo is organized to be reproducible and easy to navigate for final deliverable review.

## Research Questions
1. Can combining text embeddings with speaker-graph context improve low-truth detection?
2. Do gains remain under speaker-disjoint evaluation (less memorization risk)?
3. Which modeling strategy performs best in practice (benchmark models vs transformer fine-tuning vs tuned fusion vs ensemble)?

## Project Video
- YouTube link (2-minute project ad): **[Here](https://www.youtube.com/watch?v=sFUm1Elq3Nc)**

## Data
### LIAR dataset
- Source: PolitiFact fact-checking archive
- Dataset paper: Wang, 2017 (“Liar, Liar Pants on Fire”)
- Local files used:
  - `data/raw/train.tsv`
  - `data/raw/valid.tsv`
  - `data/raw/test.tsv`

### Preprocessing performed
- Normalize/clean key metadata fields (state, party groups).
- Build binary target `is_low_truth` from LIAR truth labels.
- Generate text features:
  - TF-IDF
  - Sentence embeddings (MiniLM, BERT)
- Build speaker-subject graph embeddings.
- Save intermediate/final artifacts in `data/interim` and `data/processed`.

## How to Reproduce
### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

### 2) Run core pipeline
```bash
bash scripts/run_pipeline.sh
```

Windows:
```powershell
scripts\run_pipeline.bat
```

### 3) Open final notebook
- Run `main_notebook.ipynb` for the curated final narrative.

## Key Dependencies
- Python 3.11
- pandas>=2.0
- numpy>=1.24
- scikit-learn>=1.3
- matplotlib>=3.7
- seaborn>=0.12
- networkx>=3.1
- pyarrow>=14.0
- sentence-transformers>=3.0
- transformers>=4.40
- torch>=2.2
- accelerate>=1.1.0

## Repository Structure
```text
.
├── main_notebook.ipynb
├── README.md
├── requirements.txt
├── checkpoints/
│   ├── checkpoint_1.ipynb
│   └── checkpoint_2.ipynb
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_patterns_graph_text.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_stream_and_drift.ipynb
│   └── question_and_hypotheses.ipynb
├── scripts/
│   ├── 01_prepare_liar.py
│   ├── ...
│   ├── 13_probability_ensemble.py
│   ├── run_pipeline.sh
│   └── run_pipeline.bat
├── src/liar_mining/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── reports/figures/
└── assets/
```

## Results Summary
- Fusion of text + graph context improves robustness over text-only models on average.
- Best benchmark results come from BERT-based embeddings with linear/logistic models under speaker-disjoint evaluation.
- Tuned tabular fusion model achieved the strongest final performance in this repo state.
- Ensemble threshold optimization provides a practical operating point for deployment-style triage.

## Notes
- Checkpoint notebooks are preserved under `checkpoints/`.
- Supporting notebooks/scripts are included for reproducibility.
- Large generated artifacts in `data/interim` and `data/processed` may be regenerated from scripts.
