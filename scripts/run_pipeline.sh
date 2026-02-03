#!/bin/bash
# Pipeline script for Edo-Meiji polysemy analysis
# This script runs the complete analysis pipeline from raw texts to statistical comparison

set -e  # Exit on error

echo "=========================================="
echo "Edo-Meiji Polysemy Analysis Pipeline"
echo "=========================================="
echo ""

# Configuration
EDO_DIR="${EDO_DIR:-data/edo}"
MEIJI_DIR="${MEIJI_DIR:-data/meiji}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-data/embeddings}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs}"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Step 1: Preprocess texts
echo "Step 1: Preprocessing texts..."
python src/data_preprocess.py \
    --edo-dir "$EDO_DIR" \
    --meiji-dir "$MEIJI_DIR" \
    --output "$PROCESSED_DIR" \
    --top-n 50 \
    --min-freq 10 \
    --max-contexts 200 \
    --log "$LOG_DIR/preprocess.log"

if [ $? -eq 0 ]; then
    echo "✓ Preprocessing complete"
else
    echo "✗ Preprocessing failed"
    exit 1
fi
echo ""

# Step 2: Extract embeddings
echo "Step 2: Extracting BERT embeddings..."
python src/embedding_extraction.py \
    --input "$PROCESSED_DIR" \
    --output "$EMBEDDINGS_DIR" \
    --model cl-tohoku/bert-base-japanese \
    --batch-size 32 \
    --device auto \
    --log "$LOG_DIR/embeddings.log"

if [ $? -eq 0 ]; then
    echo "✓ Embedding extraction complete"
else
    echo "✗ Embedding extraction failed"
    exit 1
fi
echo ""

# Step 3: Cluster embeddings and compute polysemy
echo "Step 3: Clustering embeddings and computing polysemy indices..."
python src/polysemy_clustering.py \
    --input "$EMBEDDINGS_DIR" \
    --output "$RESULTS_DIR" \
    --min-contexts 10 \
    --log "$LOG_DIR/clustering.log"

if [ $? -eq 0 ]; then
    echo "✓ Clustering and polysemy analysis complete"
else
    echo "✗ Clustering failed"
    exit 1
fi
echo ""

# Step 4: Compare eras
echo "Step 4: Comparing Edo and Meiji eras statistically..."
python src/compare_eras.py \
    --input "$RESULTS_DIR" \
    --output "$RESULTS_DIR" \
    --alpha 0.05 \
    --log "$LOG_DIR/comparison.log"

if [ $? -eq 0 ]; then
    echo "✓ Era comparison complete"
else
    echo "✗ Era comparison failed"
    exit 1
fi
echo ""

# Summary
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Output files:"
echo "  - ${RESULTS_DIR}/edo_polysemy_scores.csv"
echo "  - ${RESULTS_DIR}/meiji_polysemy_scores.csv"
echo "  - ${RESULTS_DIR}/statistical_comparison.json"
echo "  - ${RESULTS_DIR}/word_level_comparison.csv"
echo "  - ${RESULTS_DIR}/*.png (visualizations)"
echo ""
echo "To view results interactively:"
echo "  jupyter notebook notebooks/exploratory_analysis.ipynb"
echo ""
