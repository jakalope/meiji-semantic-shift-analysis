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

### Jupyter Notebook

Explore results interactively:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

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
│   └── exploratory_analysis.ipynb  # Jupyter notebook for visualization
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
