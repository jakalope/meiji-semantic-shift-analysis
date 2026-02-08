# Usage Guide

## Quick Start

This guide provides step-by-step instructions for running the Edo-Meiji polysemy analysis pipeline.

## Prerequisites

1. **Python Environment**
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   - Download Japanese texts from [Aozora Bunko](https://www.aozora.gr.jp/) or other sources
   - Organize texts by era:
     - Edo period texts → `data/edo/`
     - Meiji period texts → `data/meiji/`
   - See `data/README.md` for more details

## Running the Pipeline

### Option 1: Full Pipeline (Recommended)

Run the complete analysis with one command:

```bash
bash scripts/run_pipeline.sh
```

This will:
1. Preprocess texts and extract word contexts
2. Generate BERT embeddings
3. Cluster embeddings and compute polysemy scores
4. Compare eras statistically
5. Generate visualizations

### Option 2: Step-by-Step Execution

Run each step individually for more control:

#### Step 1: Preprocess Texts

```bash
python src/data_preprocess.py \
    --edo-dir data/edo \
    --meiji-dir data/meiji \
    --output data/processed \
    --top-n 50 \
    --min-freq 10 \
    --max-contexts 200
```

**Parameters:**
- `--edo-dir`: Directory containing Edo period texts
- `--meiji-dir`: Directory containing Meiji period texts
- `--output`: Output directory for processed data
- `--top-n`: Number of most frequent words to analyze (default: 50)
- `--min-freq`: Minimum word frequency threshold (default: 10)
- `--max-contexts`: Maximum contexts per word (default: 200)

**Output:**
- `data/processed/edo_word_frequencies.csv`
- `data/processed/edo_contexts.json`
- `data/processed/meiji_word_frequencies.csv`
- `data/processed/meiji_contexts.json`

#### Step 2: Extract BERT Embeddings

```bash
python src/embedding_extraction.py \
    --input data/processed \
    --output data/embeddings \
    --model cl-tohoku/bert-base-japanese \
    --batch-size 32 \
    --device auto
```

**Parameters:**
- `--input`: Directory with preprocessed contexts
- `--output`: Output directory for embeddings
- `--model`: HuggingFace model identifier
- `--batch-size`: Batch size for processing (adjust based on GPU memory)
- `--device`: Device to use (`auto`, `cuda`, or `cpu`)

**Output:**
- `data/embeddings/edo_embeddings.pkl`
- `data/embeddings/meiji_embeddings.pkl`

**Note:** First run will download the BERT model (~400MB). This may take several minutes depending on your internet connection.

#### Step 3: Cluster Embeddings & Compute Polysemy

```bash
python src/polysemy_clustering.py \
    --input data/embeddings \
    --output results \
    --min-contexts 10
```

**Parameters:**
- `--input`: Directory with embeddings
- `--output`: Output directory for results
- `--min-contexts`: Minimum contexts required for analysis

**Output:**
- `results/edo_polysemy_scores.csv`
- `results/meiji_polysemy_scores.csv`

#### Step 4: Compare Eras

```bash
python src/compare_eras.py \
    --input results \
    --output results \
    --alpha 0.05
```

**Parameters:**
- `--input`: Directory with polysemy scores
- `--output`: Output directory for comparison results
- `--alpha`: Significance level for statistical tests

**Output:**
- `results/statistical_comparison.json`
- `results/word_level_comparison.csv`
- `results/polysemy_distribution.png`
- `results/polysemy_boxplot.png`
- `results/cluster_comparison.png`
- `results/top_polysemy_changes.png`

## Interactive Analysis

### Local Jupyter Notebook

Launch the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

The notebook provides:
- Data loading and visualization
- Statistical analysis
- Custom plots and comparisons
- Case studies of specific words

### Google Colab (Recommended for Quick Start)

Run the complete pipeline in Google Colab without local setup:

1. **Open the notebook**:
   - Navigate to [notebooks/google_colab_pipeline.ipynb](notebooks/google_colab_pipeline.ipynb)
   - Click "Open in Colab" button
   - Or use: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakalope/meiji-semantic-shift-analysis/blob/main/notebooks/google_colab_pipeline.ipynb)

2. **Run the pipeline**:
   - Execute cells in order (Runtime > Run all)
   - First run will download BERT model (~400MB)
   - Results are saved to your Google Drive

3. **Benefits**:
   - No local installation required
   - Free GPU access for faster processing
   - Automatic Python path setup (fixes import issues)
   - Persistent storage in Google Drive
   - Runs on sample data out-of-the-box

**Note**: The Colab notebook automatically handles the Python path issue by adding `src/` to `sys.path`, eliminating the need for manual PYTHONPATH configuration.

## Interpreting Results

### Polysemy Index

The polysemy index combines:
- **Number of clusters**: More clusters suggest more distinct senses
- **Silhouette score**: Higher scores indicate better-separated clusters

Higher polysemy index = more polysemous (multiple meanings)

### Statistical Tests

- **T-test**: Parametric test comparing mean polysemy scores
- **Mann-Whitney U**: Non-parametric alternative (more robust)
- **Cohen's d**: Effect size measure
  - < 0.2: negligible
  - 0.2-0.5: small
  - 0.5-0.8: medium
  - > 0.8: large

### Visualization Files

1. `polysemy_distribution.png`: Histogram showing polysemy distribution for each era
2. `polysemy_boxplot.png`: Box plot comparison
3. `cluster_comparison.png`: Scatter plot of cluster counts (Edo vs Meiji)
4. `top_polysemy_changes.png`: Bar chart of words with largest changes

## Troubleshooting

### Out of Memory Errors

If you encounter memory errors during embedding extraction:
1. Reduce `--batch-size` (try 16 or 8)
2. Use CPU instead of GPU: `--device cpu`
3. Reduce `--max-contexts` in preprocessing

### MeCab Installation Issues

If MeCab fails to install:
```bash
# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8

# macOS
brew install mecab mecab-ipadic
```

Then install Python package:
```bash
pip install mecab-python3
```

### CUDA Not Available

If you have a GPU but CUDA is not detected:
1. Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Install CUDA-enabled PyTorch: Visit [pytorch.org](https://pytorch.org) for instructions

## Advanced Usage

### Custom Word Lists

To analyze specific words instead of top-N frequent:

1. Modify `data_preprocess.py` to load a custom word list
2. Replace the `top_words` selection with your list

### Different BERT Models

Try alternative Japanese BERT models:
- `cl-tohoku/bert-base-japanese-v3` (newer version)
- `cl-tohoku/bert-base-japanese-whole-word-masking`
- `nlp-waseda/roberta-base-japanese`

### Custom Clustering Parameters

Modify clustering in `polysemy_clustering.py`:
- Adjust `min_clusters` and `max_clusters`
- Try DBSCAN instead of K-means
- Experiment with different distance metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@software{meiji_semantic_shift,
  author = {Askeland, Jacob},
  title = {Edo-Meiji Polysemy Analysis with Japanese BERT},
  year = {2026},
  url = {https://github.com/jakalope/meiji-semantic-shift-analysis}
}
```

## Support

For issues or questions:
1. Check existing [GitHub Issues](https://github.com/jakalope/meiji-semantic-shift-analysis/issues)
2. Open a new issue with detailed description
3. Include error messages and system information
