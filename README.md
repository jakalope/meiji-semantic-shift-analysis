# Edo-Meiji Polysemy Analysis with Japanese BERT

Diachronic analysis of polysemy and semantic nuance in Edo-period vs. Meiji-era Japanese literature using BERT contextual embeddings.

## Overview

This repository implements a computational pipeline to quantify changes in word polysemy (number of distinct senses/meanings) between Edo (1603–1868) and Meiji (1868–1912) Japanese literary texts. We extract contextual embeddings from a Japanese BERT model, cluster them per word to estimate sense diversity, and compare the two eras statistically. The focus is on high-frequency content words in fiction/prose from digitized corpora.

**Research Question**: Did increased Western contact during the Meiji era lead to higher or lower polysemy (more nuanced or more specialized meanings) in literary Japanese?

## Research Motivation

The transition from the Edo period to the Meiji era represents one of the most significant cultural and linguistic shifts in Japanese history. During the Edo period, Japan maintained a policy of relative isolation (sakoku), while the Meiji Restoration (1868) opened Japan to extensive Western influence and modernization. This project uses computational methods to investigate how this transition affected semantic complexity in literary language.

Traditional linguistic analysis of historical texts is time-consuming and limited in scale. By leveraging modern NLP techniques—specifically contextual embeddings from Japanese BERT models—we can analyze semantic patterns across thousands of texts, providing quantitative evidence for diachronic semantic changes.

## Methodology

1. **Text Collection**: Gather literary texts from Edo and Meiji periods from digitized corpora (Aozora Bunko, Balanced Corpus of Historical Japanese)
2. **Preprocessing**: Tokenize texts using MeCab, filter for high-frequency content words, extract sentences containing target words
3. **Embedding Extraction**: Use cl-tohoku/bert-base-japanese to generate contextual embeddings for each occurrence of target words
4. **Clustering Analysis**: Apply K-means or DBSCAN clustering to embeddings, using silhouette scores and elbow method to estimate optimal cluster count
5. **Polysemy Index**: Calculate polysemy metrics (cluster count, silhouette score, inter-cluster distance) per word per era
6. **Statistical Comparison**: Compare distributions across eras using t-tests or Mann-Whitney U tests

## Data Sources

- **Aozora Bunko** (青空文庫): Public domain Japanese literature repository
- **Balanced Corpus of Historical Japanese**: Structured corpus of historical texts
- Sample authors:
  - Edo period: Ihara Saikaku (井原西鶴), Jippensha Ikku (十返舎一九)
  - Meiji period: Natsume Sōseki (夏目漱石), Mori Ōgai (森鷗外)

## Installation

### Requirements
- Python 3.10 or higher
- CUDA-compatible GPU (recommended, but CPU also supported)

### Setup

```bash
# Clone the repository
git clone https://github.com/jakalope/meiji-semantic-shift-analysis.git
cd meiji-semantic-shift-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Japanese BERT model (first run will automatically download)
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese'); BertModel.from_pretrained('cl-tohoku/bert-base-japanese')"
```

## Usage

### Quick Start

Run the full pipeline:
```bash
bash scripts/run_pipeline.sh
```

### Step-by-Step Execution

```python
# 1. Preprocess texts
python src/data_preprocess.py --edo-dir data/edo --meiji-dir data/meiji --output data/processed

# 2. Extract embeddings
python src/embedding_extraction.py --input data/processed --output data/embeddings

# 3. Cluster and calculate polysemy
python src/polysemy_clustering.py --input data/embeddings --output results/polysemy_scores.csv

# 4. Compare eras statistically
python src/compare_eras.py --input results/polysemy_scores.csv --output results/comparison.json
```

### Jupyter Notebooks

Explore results interactively:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Google Colab

Run the complete pipeline in Google Colab with GPU acceleration:

1. Open [google_colab_pipeline.ipynb](notebooks/google_colab_pipeline.ipynb) in Google Colab
2. Or directly: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakalope/meiji-semantic-shift-analysis/blob/main/notebooks/google_colab_pipeline.ipynb)
3. Follow the step-by-step instructions in the notebook

**Features:**
- Automatic dependency installation
- Google Drive integration for persistence
- Fixes Python path issues (no need to manually add `src/` to path)
- Runs on sample data out-of-the-box
- Includes result visualization

## Project Structure

```
meiji-semantic-shift-analysis/
├── README.md              # This file
├── LICENSE                # MIT License
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore patterns
├── src/
│   ├── utils.py          # Logging, device management, helpers
│   ├── data_preprocess.py      # Tokenization, normalization, sampling
│   ├── embedding_extraction.py # BERT model loading, embedding extraction
│   ├── polysemy_clustering.py  # Clustering, polysemy index calculation
│   └── compare_eras.py         # Statistical comparison between eras
├── data/                  # Data files (gitignored)
│   ├── README.md         # Data documentation
│   ├── edo/              # Edo period texts
│   └── meiji/            # Meiji period texts
├── notebooks/
│   ├── exploratory_analysis.ipynb  # Interactive result exploration
│   └── google_colab_pipeline.ipynb # Complete pipeline for Google Colab
├── scripts/
│   └── run_pipeline.sh   # Bash script to run full workflow
└── results/              # Output files (gitignored)
```

## Example Results

The pipeline produces:
- Polysemy scores (cluster counts) for each target word in each era
- Visualization: histograms comparing polysemy distributions
- Statistical test results (t-test or Mann-Whitney U)
- t-SNE plots of embeddings for selected words showing semantic clustering

## Methodological Considerations

### Data Quality and Normalization

**Orthographic Variation**: Edo-period texts often use historical kana orthography (旧仮名遣い kyū kanazukai) and variant kanji forms (異体字 itaiji). For accurate analysis:
- Consider normalizing historical orthography to modern equivalents before tokenization
- Document any normalization decisions as they may affect polysemy detection
- Be aware that inconsistent normalization could artifactually inflate polysemy differences

**Loanword Status**: Many Meiji semantic expansions come from new compound words rather than polysemy of native words:
- Flag gairaigo (外来語 loanwords) separately in analysis
- Focus on yamato kotoba (native Japanese) and kango (Sino-Japanese) established before 1600
- Track whether apparent polysemy increases are from new compounds vs. sense expansion

### Genre and Style Effects

**Genre Distribution**: Different literary genres may confound results:
- Edo corpus: gesaku (戯作 playful fiction), kabuki (歌舞伎), haikai (俳諧 poetry)
- Meiji corpus: shōsetsu (小説 modern novel), journalism, translated works
- **Recommendation**: Consider stratified analysis by genre or balanced sampling

**Register Changes**: The genbun itchi (言文一致) movement unified written and spoken Japanese:
- Apparent polysemy changes may reflect stylistic shifts rather than semantic expansion
- Classical literary Japanese vs. modern written style can affect word usage patterns

### Metric Validation

**Polysemy Index**: The composite metric `polysemy_index = n_clusters × (0.5 + 0.5 × normalized_silhouette)` is a practical heuristic, but:
- The 0.5 weighting is arbitrary; consider alternative metrics
- Validate against manual sense annotations on a small subset
- Compare with dictionary-based sense counts (see `data/TARGET_WORDS_METADATA.md`)

**Alternative Approaches**:
- Entropy over soft cluster assignments
- DBSCAN for density-based clustering
- Alignment-based shift scores from diachronic word embeddings

### Sensitivity Analysis

**Era Boundaries**: Semantic shifts were gradual; consider:
- Sliding window analysis (e.g., 1800-1850 vs. 1870-1920)
- Narrower sub-periods (late Edo vs. early Meiji)
- Author birth cohort as a control variable

**Clustering Parameters**:
- Current range: k=2 to k=10 for K-means
- Test with expanded range or different algorithms
- Experiment with distance metrics (cosine vs. Euclidean)

## Validation and Testing

Quick validation with sample data:
```bash
python scripts/validate_pipeline.py
```

This checks:
- Pipeline component integrity
- Target word metadata
- Sample text files
- Documentation completeness

For detailed validation guidelines, see `data/TARGET_WORDS_METADATA.md`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Japanese BERT model: [cl-tohoku/bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese) by Tohoku University
- Text sources: Aozora Bunko contributors
- Inspiration from diachronic semantics research in computational linguistics

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.
