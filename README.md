# LIAR Dataset - Data Mining Project

## Overview

This project explores the LIAR dataset, a collection of 12,791 political statements labeled with six levels of truthfulness from the PolitiFact fact-checking archive. The goal is to apply core data mining techniques while extending analysis with modern NLP and representation learning.

Research question:

**How can combining text embeddings and speaker-graph embeddings improve detection of low-truth statements in LIAR, while controlling for speaker memorization bias?**

## Dataset

Source: PolitiFact Fact-Checking Archive  
Introduced in: Wang, W. Y. (2017). “Liar, Liar Pants on Fire”

Primary LIAR files:

- `data/raw/train.tsv`
- `data/raw/valid.tsv`
- `data/raw/test.tsv`
- `data/processed/processed_liar.csv`

## Repository Structure

- `configs/`: central pipeline configuration (`configs/liar_research.yaml`)
- `scripts/`: runnable analysis pipeline scripts (`01` to `09`)
  - advanced modeling scripts: `10`, `11`, `12`, `13`
- `src/liar_mining/`: shared utilities for I/O, preprocessing, and modeling
- `notebooks/`: EDA and research analysis notebooks
- `data/interim/`: intermediate generated artifacts
- `data/processed/`: processed tables and metrics
- `models/`: model/vectorizer artifacts
- `reports/figures/`: exported plots

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_liar_research.txt
```

## Pipeline Execution Order

Run from the project root:

```bash
python scripts/01_prepare_liar.py
python scripts/02_text_embeddings.py --method tfidf
python scripts/03_graph_embeddings.py
python scripts/04_frequent_itemsets.py
python scripts/05_clustering.py --embedding-source text
python scripts/05_clustering.py --embedding-source fused
python scripts/06_lsh_near_duplicates.py
python scripts/07_anomaly_detection.py
python scripts/08_stream_drift.py
python scripts/09_fusion_model.py
```

Or run:

```bash
bash scripts/run_pipeline.sh
```

Optional dense embeddings:

```bash
python scripts/02_text_embeddings.py --method sentence --model all-MiniLM-L6-v2
```

Embedding + classifier benchmark (TF-IDF vs MiniLM vs BERT, across multiple classifiers):

```bash
python scripts/10_embedding_benchmark.py
```

Advanced additions to directly strengthen the research question answer:

```bash
# 1) Fine-tuned transformer text classifier
python scripts/11_finetune_transformer.py --model-name bert-base-uncased

# 2) Tuned tabular fused model (text + graph + metadata)
python scripts/12_tuned_tabular_fusion.py --text-file text_embeddings_sentence_bert-base-uncased.parquet

# 3) Ensemble + threshold optimization across model predictions
python scripts/13_probability_ensemble.py
```

## Techniques Used

- Frequent Itemset Mining: Apriori + association rules
- Graph Mining: speaker-subject bipartite graph
- Graph Embeddings: Node2Vec speaker/subject representations
- Text Mining: TF-IDF or sentence embeddings
- Clustering: KMeans on text/graph/fused embeddings
- LSH: MinHash near-duplicate detection
- Anomaly Detection: Isolation Forest on fused features
- Stream Mining: split-based drift analysis
- Predictive Modeling: text-only vs text+graph low-truth classification

## Key Outputs

- `data/processed/liar_base.parquet`
- `data/interim/text_embeddings_*.parquet`
- `data/interim/graph_embeddings.parquet`
- `data/processed/itemsets_*.csv` and `data/processed/rules_*.csv`
- `data/processed/clusters_*.parquet`
- `data/processed/clusters_*_quality_metrics.json`
- `data/processed/clusters_*_projection.parquet`
- `data/processed/near_duplicate_pairs.csv`
- `data/processed/near_duplicate_pairs_filtered.csv`
- `data/processed/anomaly_claims.parquet`
- `data/processed/anomaly_speakers_min_claims.csv`
- `data/processed/stream_drift_summary.csv`
- `data/processed/fusion_model_metrics.json`
- `data/processed/embedding_classifier_model_comparison.csv`
- `data/processed/transformer_finetune_metrics_*.json`
- `data/processed/tuned_tabular_fusion_metrics_*.json`
- `data/processed/ensemble_threshold_metrics.json`
- `data/processed/fusion_model_confusion_matrices.csv`
- `data/processed/fusion_model_curve_points.csv`
- `data/processed/fusion_model_threshold_sweep.csv`
- `reports/figures/clusters_*_by_*.png`

## Notes

- The pipeline is modular; each script can run independently.
- Existing original project files are preserved.

