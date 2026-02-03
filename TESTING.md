# Testing Guide

## Overview

This document describes how to test the Edo-Meiji polysemy analysis pipeline.

## Unit Testing

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_utils.py
```

## Integration Testing

### Minimal Test Data

For quick testing, create minimal sample data:

```bash
# Create test data directories
mkdir -p data/test/edo data/test/meiji

# Create sample Edo text
cat > data/test/edo/sample1.txt << 'SAMPLE'
これは江戸時代のサンプルテキストです。
人々は日々の生活を送っていました。
文化は豊かで、様々な芸術が栄えました。
SAMPLE

# Create sample Meiji text
cat > data/test/meiji/sample1.txt << 'SAMPLE'
明治時代は近代化の時代でした。
人々は新しい技術を学びました。
文化は西洋の影響を受けて変化しました。
SAMPLE
```

### Run Pipeline on Test Data

```bash
# Preprocess
python src/data_preprocess.py \
    --edo-dir data/test/edo \
    --meiji-dir data/test/meiji \
    --output data/test/processed \
    --top-n 5 \
    --min-freq 1 \
    --max-contexts 10

# Extract embeddings (requires transformers and torch)
python src/embedding_extraction.py \
    --input data/test/processed \
    --output data/test/embeddings \
    --batch-size 2

# Cluster
python src/polysemy_clustering.py \
    --input data/test/embeddings \
    --output data/test/results \
    --min-contexts 1

# Compare
python src/compare_eras.py \
    --input data/test/results \
    --output data/test/results
```

## Manual Testing Checklist

### 1. Installation
- [ ] Python 3.10+ installed
- [ ] All dependencies install without errors
- [ ] MeCab installs successfully
- [ ] BERT model downloads successfully

### 2. Preprocessing
- [ ] Tokenization works on sample text
- [ ] Word frequencies are calculated correctly
- [ ] Contexts are extracted properly
- [ ] Output files are created

### 3. Embedding Extraction
- [ ] BERT model loads without errors
- [ ] Embeddings are generated for contexts
- [ ] Batch processing works correctly
- [ ] GPU/CPU detection works

### 4. Clustering
- [ ] Embeddings cluster successfully
- [ ] Optimal cluster count is determined
- [ ] Polysemy scores are calculated
- [ ] Results are saved correctly

### 5. Comparison
- [ ] Statistical tests run without errors
- [ ] Visualizations are generated
- [ ] Results are interpretable
- [ ] Output files are created

### 6. Jupyter Notebook
- [ ] Notebook opens without errors
- [ ] All cells execute successfully
- [ ] Visualizations render correctly
- [ ] Results load properly

## Common Issues and Solutions

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'X'`
**Solution**: Install missing dependency: `pip install X`

### MeCab Errors

**Issue**: `RuntimeError: Cannot create tagger`
**Solution**: 
1. Install system MeCab: `sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8`
2. Reinstall Python package: `pip install --force-reinstall mecab-python3`

### Memory Errors

**Issue**: `CUDA out of memory`
**Solution**:
1. Reduce batch size: `--batch-size 8`
2. Use CPU: `--device cpu`
3. Reduce number of contexts: `--max-contexts 50`

### Empty Results

**Issue**: No words analyzed or empty output
**Solution**:
1. Check input data exists and is readable
2. Verify text encoding is UTF-8
3. Reduce `--min-freq` threshold
4. Check log files for warnings

## Performance Benchmarks

Expected processing times (approximate, will vary by system):

| Step | Small Dataset (10 texts) | Medium Dataset (100 texts) |
|------|-------------------------|---------------------------|
| Preprocessing | < 1 min | 5-10 min |
| Embedding Extraction (GPU) | 2-5 min | 20-40 min |
| Embedding Extraction (CPU) | 10-20 min | 1-2 hours |
| Clustering | < 1 min | 2-5 min |
| Comparison | < 1 min | < 1 min |

## Continuous Integration

For automated testing in CI/CD:

```yaml
# Example .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=src tests/
```

## Reporting Bugs

When reporting issues, include:
1. Python version: `python --version`
2. Dependency versions: `pip freeze`
3. Error message and full traceback
4. Steps to reproduce
5. Sample data (if applicable)
6. Log files

Open issues at: https://github.com/jakalope/meiji-semantic-shift-analysis/issues
