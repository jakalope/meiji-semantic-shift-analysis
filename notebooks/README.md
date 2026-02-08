# Notebooks

This directory contains Jupyter notebooks for the Edo-Meiji polysemy analysis project.

## Available Notebooks

### 1. meiji-download-to-drive.ipynb

**Purpose**: Download Meiji period texts from Aozora Bunko to Google Drive

**Best for**: 
- Users who want to analyze full Meiji period texts (not just samples)
- Preparing data for the main analysis pipeline

**Features**:
- ✅ Downloads ~32M characters of Meiji period texts from Aozora Bunko
- ✅ Filters texts by Meiji era criteria (authors active/died during Meiji)
- ✅ Saves to Google Drive at `/content/drive/MyDrive/meiji-semantic-data/meiji/`
- ✅ One-time download, reusable across sessions

**How to use**:
1. Open in Colab: [meiji-download-to-drive.ipynb](https://colab.research.google.com/github/jakalope/meiji-semantic-shift-analysis/blob/main/notebooks/meiji-download-to-drive.ipynb)
2. Run all cells to download and save Meiji texts
3. Then use `google_colab_pipeline.ipynb` which will automatically detect and use this data

**Runtime**: ~5-10 minutes (one-time download)

---

### 2. google_colab_pipeline.ipynb

**Purpose**: Complete end-to-end pipeline for Google Colab

**Best for**: 
- First-time users wanting to try the pipeline quickly
- Users without local GPU
- Running analysis without local setup

**Features**:
- ✅ Runs entirely in Google Colab (no local installation needed)
- ✅ Automatic Google Drive mounting for persistence
- ✅ Repository cloning and dependency installation
- ✅ Fixes Python path issues automatically (adds `src/` to `sys.path`)
- ✅ **Automatically detects and uses Meiji data from meiji-download-to-drive.ipynb**
- ✅ Falls back to sample data if full data not available
- ✅ Complete pipeline execution (preprocess → embeddings → clustering → comparison)
- ✅ Built-in result visualization
- ✅ Free GPU acceleration available

**Recommended workflow**:
1. (Optional) First run [meiji-download-to-drive.ipynb](https://colab.research.google.com/github/jakalope/meiji-semantic-shift-analysis/blob/main/notebooks/meiji-download-to-drive.ipynb) to get full Meiji data
2. Then run this notebook - it automatically detects downloaded data

**How to use**:
1. Click to open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakalope/meiji-semantic-shift-analysis/blob/main/notebooks/google_colab_pipeline.ipynb)
2. Run cells in order (or Runtime > Run all)
3. Authorize Google Drive access when prompted
4. Results are saved to your Google Drive

**Runtime**: 
- With sample data + GPU: ~5-10 minutes (first run downloads ~400MB BERT model)
- With sample data + CPU: ~15-20 minutes
- With full downloaded data: Significantly longer (depends on data size)

---

### 3. exploratory_analysis.ipynb

**Purpose**: Interactive exploration of analysis results

**Best for**:
- Users who have already run the pipeline
- Custom analysis and visualization
- Deep dive into specific words or patterns
- Generating custom plots and reports

**Features**:
- Load and explore existing results
- Custom visualizations
- Statistical analysis
- Word-level case studies
- Flexible data manipulation

**Requirements**:
- Local Jupyter installation
- Results from running the pipeline (either via Colab or local execution)

**How to use**:
```bash
# From repository root
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## Which Notebook Should I Use?

| Scenario | Recommended Notebook |
|----------|---------------------|
| Want to download full Meiji data | `meiji-download-to-drive.ipynb` (do this first) |
| First time trying the project | `google_colab_pipeline.ipynb` |
| No local Python/GPU setup | `google_colab_pipeline.ipynb` |
| Want to run full pipeline quickly | `google_colab_pipeline.ipynb` |
| Already have results, want to explore | `exploratory_analysis.ipynb` |
| Custom analysis on existing data | `exploratory_analysis.ipynb` |
| Local development environment | `exploratory_analysis.ipynb` |

---

## Python Path Issue (Fixed in Colab Notebook)

**Problem**: Running scripts that import from `src/` requires adding the `src/` directory to Python path.

**Solution in Colab**: The `google_colab_pipeline.ipynb` notebook automatically fixes this by running:

```python
import sys
import os

# Add src directory to Python path
SRC_PATH = os.path.join(os.getcwd(), 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
```

This allows imports like `import utils`, `import data_preprocess`, etc. to work correctly.

**Local Alternative**: If running scripts locally, you can:
1. Add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
2. Use the Colab notebook approach in your local notebook
3. Install as package: `pip install -e .` (if setup.py exists)

---

## Support

For issues or questions:
- Check the main [README.md](../README.md)
- Read [USAGE.md](../USAGE.md) for detailed instructions
- Open an issue on [GitHub](https://github.com/jakalope/meiji-semantic-shift-analysis/issues)
