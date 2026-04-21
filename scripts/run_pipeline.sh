#!/usr/bin/env bash
set -uo pipefail

PY=python

run_step() {
  step="$1"
  shift
  echo "$step"
  "$PY" "$@"
  code=$?
  if [ $code -ne 0 ]; then
    echo
    echo "Pipeline failed with exit code $code."
    exit $code
  fi
}

run_step "[1/14] Preparing LIAR base table..." scripts/01_prepare_liar.py
run_step "[2/14] Generating TF-IDF text embeddings..." scripts/02_text_embeddings.py --method tfidf
run_step "[3/14] Building graph embeddings..." scripts/03_graph_embeddings.py
run_step "[4/14] Mining frequent itemsets..." scripts/04_frequent_itemsets.py
run_step "[5/14] Running clustering (text)..." scripts/05_clustering.py --embedding-source text
run_step "[6/14] Running clustering (fused)..." scripts/05_clustering.py --embedding-source fused
run_step "[7/14] Detecting near-duplicates (LSH)..." scripts/06_lsh_near_duplicates.py
run_step "[8/14] Running anomaly detection..." scripts/07_anomaly_detection.py
run_step "[9/14] Computing stream drift..." scripts/08_stream_drift.py
run_step "[10/14] Training baseline fusion model..." scripts/09_fusion_model.py
run_step "[11/14] Running embedding + classifier benchmark..." scripts/10_embedding_benchmark.py
run_step "[12/14] Fine-tuning transformer classifier..." scripts/11_finetune_transformer.py --model-name bert-base-uncased
run_step "[13/14] Training tuned tabular fusion model..." scripts/12_tuned_tabular_fusion.py --text-file text_embeddings_sentence_bert-base-uncased.parquet
run_step "[14/14] Running ensemble + threshold optimization..." scripts/13_probability_ensemble.py

echo
echo "Pipeline complete. Outputs are in data/processed and data/interim."
