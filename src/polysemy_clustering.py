"""
Polysemy clustering module for semantic analysis.

This module handles:
- Clustering contextual embeddings using K-means and DBSCAN
- Computing optimal cluster counts using elbow method and silhouette scores
- Calculating polysemy indices
- Visualizing embedding clusters
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

from utils import setup_logging, ensure_dir, load_pickle, save_json


class PolysemyAnalyzer:
    """
    Analyze polysemy through clustering of contextual embeddings.
    """
    
    def __init__(self, min_clusters: int = 2, max_clusters: int = 10, random_state: int = 42):
        """
        Initialize polysemy analyzer.
        
        Args:
            min_clusters: Minimum number of clusters to test
            max_clusters: Maximum number of clusters to test
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.logger = logging.getLogger("edo_meiji_analysis")
    
    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        method: str = 'silhouette'
    ) -> Tuple[int, float]:
        """
        Find optimal number of clusters using elbow or silhouette method.
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            method: Method to use ('silhouette' or 'elbow')
            
        Returns:
            Tuple of (optimal_k, score)
        """
        n_samples = len(embeddings)
        
        # Adjust max_clusters based on sample size
        max_k = min(self.max_clusters, n_samples - 1)
        
        if n_samples < self.min_clusters:
            self.logger.warning(f"Not enough samples ({n_samples}) for clustering")
            return 1, 0.0
        
        scores = []
        k_range = range(self.min_clusters, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            if method == 'silhouette':
                # Higher is better
                score = silhouette_score(embeddings, labels)
            elif method == 'elbow':
                # Lower is better (inertia)
                score = -kmeans.inertia_
            else:
                raise ValueError(f"Unknown method: {method}")
            
            scores.append(score)
        
        # Find optimal k
        if method == 'silhouette':
            optimal_idx = np.argmax(scores)
        else:  # elbow
            # Use elbow detection (find point of maximum curvature)
            optimal_idx = self._find_elbow_point(scores)
        
        optimal_k = list(k_range)[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        return optimal_k, optimal_score
    
    def _find_elbow_point(self, scores: List[float]) -> int:
        """
        Find elbow point in a curve using the maximum distance method.
        
        Args:
            scores: List of scores (e.g., inertia values)
            
        Returns:
            Index of elbow point
        """
        if len(scores) < 3:
            return 0
        
        # Normalize scores to [0, 1]
        scores_array = np.array(scores)
        scores_norm = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min() + 1e-10)
        
        # Calculate distances from line connecting first and last points
        n_points = len(scores_norm)
        line_start = np.array([0, scores_norm[0]])
        line_end = np.array([n_points - 1, scores_norm[-1]])
        
        distances = []
        for i in range(n_points):
            point = np.array([i, scores_norm[i]])
            distance = self._point_line_distance(point, line_start, line_end)
            distances.append(distance)
        
        return np.argmax(distances)
    
    def _point_line_distance(
        self,
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> float:
        """Calculate perpendicular distance from point to line."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-10:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        projection = np.dot(point_vec, line_unitvec)
        nearest = line_start + projection * line_unitvec
        
        return np.linalg.norm(point - nearest)
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Cluster embeddings and compute quality metrics.
        
        Args:
            embeddings: Array of embeddings
            n_clusters: Number of clusters (if None, will be determined automatically)
            
        Returns:
            Tuple of (cluster_labels, metrics_dict)
        """
        n_samples = len(embeddings)
        
        if n_samples < 2:
            return np.zeros(n_samples), {'n_clusters': 1, 'silhouette': 0.0}
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Determine optimal clusters if not specified
        if n_clusters is None:
            n_clusters, sil_score = self.find_optimal_clusters(embeddings_scaled, method='silhouette')
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings_scaled)
        
        # Compute metrics
        metrics = {
            'n_clusters': n_clusters,
            'n_samples': n_samples
        }
        
        if n_clusters > 1:
            metrics['silhouette'] = silhouette_score(embeddings_scaled, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(embeddings_scaled, labels)
            metrics['inertia'] = kmeans.inertia_
        else:
            metrics['silhouette'] = 0.0
            metrics['davies_bouldin'] = 0.0
            metrics['inertia'] = 0.0
        
        return labels, metrics
    
    def compute_polysemy_index(self, metrics: Dict[str, float]) -> float:
        """
        Compute a polysemy index from clustering metrics.
        
        The polysemy index combines:
        - Number of clusters (more clusters = higher polysemy)
        - Silhouette score (better separation = more distinct senses)
        
        Args:
            metrics: Dictionary of clustering metrics
            
        Returns:
            Polysemy index (higher = more polysemous)
        """
        n_clusters = metrics.get('n_clusters', 1)
        silhouette = metrics.get('silhouette', 0.0)
        
        # Simple polysemy index: weighted combination
        # Normalize silhouette to [0, 1] (it's in [-1, 1])
        silhouette_norm = (silhouette + 1) / 2
        
        # Polysemy index: cluster count weighted by quality
        polysemy = n_clusters * (0.5 + 0.5 * silhouette_norm)
        
        return polysemy


def analyze_corpus_polysemy(
    embeddings_file: str,
    output_dir: str,
    min_contexts: int = 10
) -> pd.DataFrame:
    """
    Analyze polysemy for all words in a corpus.
    
    Args:
        embeddings_file: Path to embeddings pickle file
        output_dir: Output directory
        min_contexts: Minimum number of contexts required for analysis
        
    Returns:
        DataFrame with polysemy metrics for each word
    """
    logger = logging.getLogger("edo_meiji_analysis")
    logger.info(f"Analyzing polysemy from {embeddings_file}")
    
    # Load embeddings
    embeddings_data = load_pickle(embeddings_file)
    
    # Determine era from filename
    era = os.path.basename(embeddings_file).split('_')[0]
    
    # Initialize analyzer
    analyzer = PolysemyAnalyzer()
    
    # Analyze each word
    results = []
    
    for word, data in tqdm(embeddings_data.items(), desc=f"Analyzing {era} words"):
        embeddings = data['embeddings']
        n_contexts = data['n_contexts']
        
        if n_contexts < min_contexts:
            logger.warning(f"Word '{word}' has only {n_contexts} contexts, skipping")
            continue
        
        # Cluster embeddings
        labels, metrics = analyzer.cluster_embeddings(embeddings)
        
        # Compute polysemy index
        polysemy_index = analyzer.compute_polysemy_index(metrics)
        
        # Store results
        result = {
            'word': word,
            'era': era,
            'n_contexts': n_contexts,
            'n_clusters': metrics['n_clusters'],
            'silhouette': metrics['silhouette'],
            'davies_bouldin': metrics.get('davies_bouldin', 0.0),
            'polysemy_index': polysemy_index
        }
        results.append(result)
        
        logger.debug(f"Word '{word}': {metrics['n_clusters']} clusters, "
                    f"silhouette={metrics['silhouette']:.3f}, "
                    f"polysemy={polysemy_index:.3f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    ensure_dir(output_dir)
    output_file = os.path.join(output_dir, f'{era}_polysemy_scores.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"Saved polysemy scores to {output_file}")
    
    return df


def main():
    """Main entry point for polysemy clustering."""
    parser = argparse.ArgumentParser(description="Cluster embeddings and compute polysemy indices")
    parser.add_argument('--input', type=str, required=True, help='Input directory with embeddings')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--min-contexts', type=int, default=10, 
                       help='Minimum contexts required for analysis')
    parser.add_argument('--log', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log)
    
    # Process both eras
    for era in ['edo', 'meiji']:
        embeddings_file = os.path.join(args.input, f'{era}_embeddings.pkl')
        
        if os.path.exists(embeddings_file):
            analyze_corpus_polysemy(embeddings_file, args.output, args.min_contexts)
        else:
            logger.warning(f"Embeddings file not found: {embeddings_file}")
    
    logger.info("Polysemy analysis complete!")


if __name__ == '__main__':
    main()
