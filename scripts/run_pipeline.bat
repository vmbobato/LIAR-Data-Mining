@echo off
setlocal enabledelayedexpansion

REM Run from project root:
REM   scripts\run_pipeline.bat

set "PY=python"

echo [1/14] Preparing LIAR base table...
%PY% scripts/01_prepare_liar.py
if errorlevel 1 goto :fail

echo [2/14] Generating TF-IDF text embeddings...
%PY% scripts/02_text_embeddings.py --method tfidf
if errorlevel 1 goto :fail

echo [3/14] Building graph embeddings...
%PY% scripts/03_graph_embeddings.py
if errorlevel 1 goto :fail

echo [4/14] Mining frequent itemsets...
%PY% scripts/04_frequent_itemsets.py
if errorlevel 1 goto :fail

echo [5/14] Running clustering (text)...
%PY% scripts/05_clustering.py --embedding-source text
if errorlevel 1 goto :fail

echo [6/14] Running clustering (fused)...
%PY% scripts/05_clustering.py --embedding-source fused
if errorlevel 1 goto :fail

echo [7/14] Detecting near-duplicates (LSH)...
%PY% scripts/06_lsh_near_duplicates.py
if errorlevel 1 goto :fail

echo [8/14] Running anomaly detection...
%PY% scripts/07_anomaly_detection.py
if errorlevel 1 goto :fail

echo [9/14] Computing stream drift...
%PY% scripts/08_stream_drift.py
if errorlevel 1 goto :fail

echo [10/14] Training baseline fusion model...
%PY% scripts/09_fusion_model.py
if errorlevel 1 goto :fail

echo [11/14] Running embedding + classifier benchmark...
%PY% scripts/10_embedding_benchmark.py
if errorlevel 1 goto :fail

echo [12/14] Fine-tuning transformer classifier...
%PY% scripts/11_finetune_transformer.py --model-name bert-base-uncased
if errorlevel 1 goto :fail

echo [13/14] Training tuned tabular fusion model...
%PY% scripts/12_tuned_tabular_fusion.py --text-file text_embeddings_sentence_bert-base-uncased.parquet
if errorlevel 1 goto :fail

echo [14/14] Running ensemble + threshold optimization...
%PY% scripts/13_probability_ensemble.py
if errorlevel 1 goto :fail

echo.
echo Pipeline complete. Outputs are in data/processed and data/interim.
exit /b 0

:fail
echo.
echo Pipeline failed with exit code %errorlevel%.
exit /b %errorlevel%
