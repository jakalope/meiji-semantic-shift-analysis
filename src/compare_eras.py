"""
Era comparison module for analyzing polysemy differences.

This module handles:
- Statistical comparison of polysemy between Edo and Meiji eras
- Computing effect sizes and significance tests
- Generating comparison visualizations
- Identifying words with significant polysemy changes
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available. Install with: pip install scipy")

from utils import setup_logging, ensure_dir, save_json


class EraComparator:
    """
    Compare polysemy metrics between Edo and Meiji eras.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize era comparator.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.logger = logging.getLogger("edo_meiji_analysis")
    
    def compare_distributions(
        self,
        edo_scores: np.ndarray,
        meiji_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare polysemy score distributions using statistical tests.
        
        Args:
            edo_scores: Polysemy scores from Edo era
            meiji_scores: Polysemy scores from Meiji era
            
        Returns:
            Dictionary of statistical test results
        """
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not available, skipping statistical tests")
            return {}
        
        results = {}
        
        # Descriptive statistics
        results['edo_mean'] = float(np.mean(edo_scores))
        results['edo_std'] = float(np.std(edo_scores))
        results['edo_median'] = float(np.median(edo_scores))
        results['meiji_mean'] = float(np.mean(meiji_scores))
        results['meiji_std'] = float(np.std(meiji_scores))
        results['meiji_median'] = float(np.median(meiji_scores))
        
        # Mean difference
        results['mean_difference'] = results['meiji_mean'] - results['edo_mean']
        results['median_difference'] = results['meiji_median'] - results['edo_median']
        
        # T-test (parametric)
        t_stat, t_pval = stats.ttest_ind(edo_scores, meiji_scores)
        results['t_statistic'] = float(t_stat)
        results['t_pvalue'] = float(t_pval)
        results['t_significant'] = t_pval < self.alpha
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(edo_scores, meiji_scores, alternative='two-sided')
        results['mannwhitney_statistic'] = float(u_stat)
        results['mannwhitney_pvalue'] = float(u_pval)
        results['mannwhitney_significant'] = u_pval < self.alpha
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.std(edo_scores)**2 + np.std(meiji_scores)**2) / 2)
        cohens_d = (results['meiji_mean'] - results['edo_mean']) / (pooled_std + 1e-10)
        results['cohens_d'] = float(cohens_d)
        
        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_size = 'negligible'
        elif abs_d < 0.5:
            effect_size = 'small'
        elif abs_d < 0.8:
            effect_size = 'medium'
        else:
            effect_size = 'large'
        results['effect_size_interpretation'] = effect_size
        
        return results
    
    def compare_word_level(
        self,
        edo_df: pd.DataFrame,
        meiji_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare polysemy scores at the word level.
        
        Args:
            edo_df: DataFrame with Edo polysemy scores
            meiji_df: DataFrame with Meiji polysemy scores
            
        Returns:
            DataFrame with word-level comparisons
        """
        # Merge on common words
        merged = edo_df.merge(
            meiji_df,
            on='word',
            suffixes=('_edo', '_meiji'),
            how='inner'
        )
        
        if len(merged) == 0:
            self.logger.warning("No common words found between eras")
            return pd.DataFrame()
        
        # Calculate differences
        merged['polysemy_change'] = merged['polysemy_index_meiji'] - merged['polysemy_index_edo']
        merged['polysemy_change_pct'] = (
            100 * merged['polysemy_change'] / (merged['polysemy_index_edo'] + 1e-10)
        )
        merged['cluster_change'] = merged['n_clusters_meiji'] - merged['n_clusters_edo']
        
        # Sort by absolute change
        merged['abs_change'] = merged['polysemy_change'].abs()
        merged = merged.sort_values('abs_change', ascending=False)
        
        return merged


def load_polysemy_scores(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load polysemy scores for both eras.
    
    Args:
        results_dir: Directory containing polysemy score CSV files
        
    Returns:
        Tuple of (edo_df, meiji_df)
    """
    logger = logging.getLogger("edo_meiji_analysis")
    
    edo_file = os.path.join(results_dir, 'edo_polysemy_scores.csv')
    meiji_file = os.path.join(results_dir, 'meiji_polysemy_scores.csv')
    
    if not os.path.exists(edo_file):
        raise FileNotFoundError(f"Edo scores not found: {edo_file}")
    if not os.path.exists(meiji_file):
        raise FileNotFoundError(f"Meiji scores not found: {meiji_file}")
    
    edo_df = pd.read_csv(edo_file)
    meiji_df = pd.read_csv(meiji_file)
    
    logger.info(f"Loaded {len(edo_df)} Edo words and {len(meiji_df)} Meiji words")
    
    return edo_df, meiji_df


def create_visualizations(
    edo_df: pd.DataFrame,
    meiji_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Create visualization plots comparing eras.
    
    Args:
        edo_df: DataFrame with Edo polysemy scores
        meiji_df: DataFrame with Meiji polysemy scores
        comparison_df: DataFrame with word-level comparisons
        output_dir: Output directory for plots
    """
    logger = logging.getLogger("edo_meiji_analysis")
    logger.info("Creating visualizations")
    
    ensure_dir(output_dir)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Distribution comparison histogram
    plt.figure(figsize=(10, 6))
    plt.hist(edo_df['polysemy_index'], bins=20, alpha=0.5, label='Edo', color='blue')
    plt.hist(meiji_df['polysemy_index'], bins=20, alpha=0.5, label='Meiji', color='red')
    plt.xlabel('Polysemy Index')
    plt.ylabel('Frequency')
    plt.title('Polysemy Index Distribution: Edo vs Meiji')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'polysemy_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Box plot comparison
    plt.figure(figsize=(8, 6))
    data_to_plot = [edo_df['polysemy_index'], meiji_df['polysemy_index']]
    plt.boxplot(data_to_plot, labels=['Edo', 'Meiji'])
    plt.ylabel('Polysemy Index')
    plt.title('Polysemy Index by Era')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'polysemy_boxplot.png'), dpi=300)
    plt.close()
    
    # 3. Cluster count comparison
    if not comparison_df.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            comparison_df['n_clusters_edo'],
            comparison_df['n_clusters_meiji'],
            alpha=0.6
        )
        max_clusters = max(comparison_df['n_clusters_edo'].max(), 
                          comparison_df['n_clusters_meiji'].max())
        plt.plot([0, max_clusters], [0, max_clusters], 'r--', alpha=0.5, label='No change')
        plt.xlabel('Edo Cluster Count')
        plt.ylabel('Meiji Cluster Count')
        plt.title('Cluster Count Comparison (Each point = one word)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_comparison.png'), dpi=300)
        plt.close()
        
        # 4. Top words with polysemy change
        top_n = min(20, len(comparison_df))
        top_changes = comparison_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), top_changes['polysemy_change'])
        plt.yticks(range(top_n), top_changes['word'])
        plt.xlabel('Polysemy Change (Meiji - Edo)')
        plt.title(f'Top {top_n} Words by Absolute Polysemy Change')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_polysemy_changes.png'), dpi=300)
        plt.close()
    
    logger.info(f"Saved visualizations to {output_dir}")


def main():
    """Main entry point for era comparison."""
    parser = argparse.ArgumentParser(description="Compare polysemy between Edo and Meiji eras")
    parser.add_argument('--input', type=str, required=True, 
                       help='Input directory with polysemy scores')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--log', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log)
    
    try:
        # Load polysemy scores
        edo_df, meiji_df = load_polysemy_scores(args.input)
        
        # Initialize comparator
        comparator = EraComparator(alpha=args.alpha)
        
        # Compare distributions
        logger.info("Comparing overall distributions")
        dist_results = comparator.compare_distributions(
            edo_df['polysemy_index'].values,
            meiji_df['polysemy_index'].values
        )
        
        # Print summary
        logger.info("=" * 60)
        logger.info("STATISTICAL COMPARISON SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Edo period: mean={dist_results['edo_mean']:.3f}, "
                   f"median={dist_results['edo_median']:.3f}")
        logger.info(f"Meiji period: mean={dist_results['meiji_mean']:.3f}, "
                   f"median={dist_results['meiji_median']:.3f}")
        logger.info(f"Mean difference: {dist_results['mean_difference']:.3f}")
        logger.info(f"Cohen's d: {dist_results['cohens_d']:.3f} "
                   f"({dist_results['effect_size_interpretation']})")
        logger.info(f"T-test p-value: {dist_results['t_pvalue']:.4f} "
                   f"({'significant' if dist_results['t_significant'] else 'not significant'})")
        logger.info(f"Mann-Whitney p-value: {dist_results['mannwhitney_pvalue']:.4f} "
                   f"({'significant' if dist_results['mannwhitney_significant'] else 'not significant'})")
        logger.info("=" * 60)
        
        # Word-level comparison
        logger.info("Performing word-level comparison")
        comparison_df = comparator.compare_word_level(edo_df, meiji_df)
        
        # Save results
        ensure_dir(args.output)
        
        # Save statistical results
        save_json(dist_results, os.path.join(args.output, 'statistical_comparison.json'))
        
        # Save word-level comparison
        if not comparison_df.empty:
            comparison_df.to_csv(
                os.path.join(args.output, 'word_level_comparison.csv'),
                index=False
            )
            logger.info(f"Found {len(comparison_df)} common words between eras")
        
        # Create visualizations
        create_visualizations(edo_df, meiji_df, comparison_df, args.output)
        
        logger.info("Era comparison complete!")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
