"""
Text preprocessing module for Edo-Meiji polysemy analysis.

This module handles:
- Japanese text tokenization using MeCab
- Text normalization
- Sentence extraction for target words
- Filtering high-frequency content words
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter
import pandas as pd
from tqdm import tqdm

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    logging.warning("MeCab not available. Install mecab-python3 for full functionality.")

from utils import setup_logging, ensure_dir, save_json, load_json


class JapaneseTokenizer:
    """
    Japanese text tokenizer using MeCab.
    """
    
    def __init__(self):
        """Initialize MeCab tokenizer."""
        if not MECAB_AVAILABLE:
            raise ImportError("MeCab is required for tokenization. Install with: pip install mecab-python3")
        
        self.tagger = self._create_mecab_tagger()
    
    def _create_mecab_tagger(self):
        """
        Create MeCab tagger with fallback configuration paths.
        
        Returns:
            MeCab.Tagger instance
            
        Raises:
            RuntimeError: If MeCab cannot be initialized
        """
        # Detect if running in Google Colab
        try:
            import google.colab
            in_colab = True
        except ImportError:
            in_colab = False
        
        # Colab fix: explicitly use system mecabrc and ipadic-utf8 dictionary path
        # mecab-python3 defaults to /usr/local/etc/mecabrc which doesn't exist on Colab
        if in_colab:
            try:
                return MeCab.Tagger('-r /etc/mecabrc -d /var/lib/mecab/dic/ipadic-utf8')
            except RuntimeError:
                # Fallback to just config file if full path fails
                try:
                    return MeCab.Tagger('-r /etc/mecabrc')
                except RuntimeError:
                    pass
        
        # Try default initialization first (for non-Colab environments)
        try:
            return MeCab.Tagger()
        except RuntimeError:
            pass
        
        # Common configuration paths (order matters - most common first)
        # Can be overridden with MECAB_CONFIG and MECAB_DICT environment variables
        config_path = os.environ.get('MECAB_CONFIG')
        dict_path = os.environ.get('MECAB_DICT')
        
        if config_path and dict_path:
            config_paths = [(config_path, dict_path)]
        else:
            config_paths = [
                ('/etc/mecabrc', '/var/lib/mecab/dic/ipadic-utf8'),
                ('/etc/mecabrc', '/usr/share/mecab/dic/ipadic'),
                ('/usr/local/etc/mecabrc', '/usr/local/lib/mecab/dic/ipadic'),
            ]
        
        # Try with config and dictionary
        for cfg, dic in config_paths:
            if os.path.exists(cfg) and os.path.exists(dic):
                try:
                    return MeCab.Tagger(f'-r {cfg} -d {dic}')
                except RuntimeError:
                    pass
        
        # Last resort: try with just dictionary path
        dict_paths = [path for _, path in config_paths if os.path.exists(path)]
        for dic in dict_paths:
            try:
                return MeCab.Tagger(f'-d {dic}')
            except RuntimeError:
                pass
        
        # All attempts failed
        raise RuntimeError(
            "Failed to initialize MeCab. Please ensure MeCab is properly installed.\n"
            "Installation instructions:\n"
            "  - Debian/Ubuntu: sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8\n"
            "  - macOS: brew install mecab mecab-ipadic\n"
            "  - Other systems: see https://taku910.github.io/mecab/\n"
            "You can also set MECAB_CONFIG and MECAB_DICT environment variables."
        )
    
    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        """
        Tokenize Japanese text and return words with their POS tags.
        
        Args:
            text: Input text
            
        Returns:
            List of (word, pos) tuples
        """
        self.tagger.parse('')  # Reset state
        node = self.tagger.parseToNode(text)
        
        tokens = []
        while node:
            if node.surface:  # Skip BOS/EOS
                features = node.feature.split(',')
                word = node.surface
                pos = features[0] if features else 'Unknown'
                tokens.append((word, pos))
            node = node.next
        
        return tokens
    
    def is_content_word(self, pos: str) -> bool:
        """
        Check if a POS tag represents a content word.
        
        Args:
            pos: Part-of-speech tag
            
        Returns:
            True if content word (noun, verb, adjective, adverb)
        """
        content_pos = ['名詞', '動詞', '形容詞', '副詞']
        return any(pos.startswith(p) for p in content_pos)


def normalize_text(text: str) -> str:
    """
    Normalize Japanese text.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (optional - preserve Japanese punctuation)
    # text = re.sub(r'[^\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\u3400-\u4dbf]', '', text)
    
    return text.strip()


def extract_sentences(text: str, max_length: int = 200) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        max_length: Maximum sentence length (in characters)
        
    Returns:
        List of sentences
    """
    # Split on Japanese sentence-ending punctuation
    sentences = re.split(r'[。！？\n]+', text)
    
    # Filter out empty and overly long sentences
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) <= max_length]
    
    return sentences


def collect_word_frequencies(
    texts: List[str],
    tokenizer: JapaneseTokenizer,
    min_freq: int = 10
) -> Counter:
    """
    Collect frequency counts for content words.
    
    Args:
        texts: List of text strings
        tokenizer: JapaneseTokenizer instance
        min_freq: Minimum frequency threshold
        
    Returns:
        Counter of word frequencies
    """
    word_counts = Counter()
    
    for text in tqdm(texts, desc="Counting word frequencies"):
        tokens = tokenizer.tokenize(text)
        content_words = [word for word, pos in tokens if tokenizer.is_content_word(pos)]
        word_counts.update(content_words)
    
    # Filter by minimum frequency
    filtered_counts = Counter({word: count for word, count in word_counts.items() if count >= min_freq})
    
    return filtered_counts


def extract_word_contexts(
    texts: List[str],
    target_words: Set[str],
    tokenizer: JapaneseTokenizer,
    max_contexts: int = 200
) -> Dict[str, List[str]]:
    """
    Extract sentences containing target words.
    
    Args:
        texts: List of text strings
        target_words: Set of words to extract contexts for
        tokenizer: JapaneseTokenizer instance
        max_contexts: Maximum number of contexts per word
        
    Returns:
        Dictionary mapping words to lists of context sentences
    """
    word_contexts = {word: [] for word in target_words}
    
    for text in tqdm(texts, desc="Extracting word contexts"):
        sentences = extract_sentences(text)
        
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            words_in_sentence = {word for word, pos in tokens}
            
            # Check which target words appear in this sentence
            for target_word in target_words:
                if target_word in words_in_sentence:
                    if len(word_contexts[target_word]) < max_contexts:
                        word_contexts[target_word].append(sentence)
    
    return word_contexts


def process_corpus(
    corpus_dir: str,
    output_dir: str,
    era_label: str,
    top_n_words: int = 50,
    min_freq: int = 10,
    max_contexts: int = 200
) -> None:
    """
    Process a corpus directory and extract word contexts.
    
    Args:
        corpus_dir: Directory containing text files
        output_dir: Output directory
        era_label: Label for the era (e.g., 'edo', 'meiji')
        top_n_words: Number of top-frequency words to analyze
        min_freq: Minimum word frequency threshold
        max_contexts: Maximum contexts per word
    """
    logger = logging.getLogger("edo_meiji_analysis")
    logger.info(f"Processing {era_label} corpus from {corpus_dir}")
    
    # Initialize tokenizer
    tokenizer = JapaneseTokenizer()
    
    # Load all texts
    texts = []
    corpus_path = Path(corpus_dir)
    for txt_file in corpus_path.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = normalize_text(f.read())
            texts.append(text)
    
    logger.info(f"Loaded {len(texts)} texts")
    
    # Collect word frequencies
    word_freqs = collect_word_frequencies(texts, tokenizer, min_freq)
    logger.info(f"Found {len(word_freqs)} words with frequency >= {min_freq}")
    
    # Select top N words
    top_words = [word for word, count in word_freqs.most_common(top_n_words)]
    logger.info(f"Selected top {len(top_words)} words for analysis")
    
    # Extract contexts
    word_contexts = extract_word_contexts(texts, set(top_words), tokenizer, max_contexts)
    
    # Save results
    ensure_dir(output_dir)
    
    # Save word frequencies
    freq_df = pd.DataFrame(list(word_freqs.items()), columns=['word', 'frequency'])
    freq_df = freq_df.sort_values('frequency', ascending=False)
    freq_df.to_csv(os.path.join(output_dir, f'{era_label}_word_frequencies.csv'), index=False)
    
    # Save contexts
    contexts_data = {
        'era': era_label,
        'words': {word: {'contexts': contexts, 'count': len(contexts)} 
                  for word, contexts in word_contexts.items()}
    }
    save_json(contexts_data, os.path.join(output_dir, f'{era_label}_contexts.json'))
    
    logger.info(f"Saved results to {output_dir}")


def main():
    """Main entry point for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess Japanese text corpus")
    parser.add_argument('--edo-dir', type=str, help='Directory containing Edo period texts')
    parser.add_argument('--meiji-dir', type=str, help='Directory containing Meiji period texts')
    parser.add_argument('--output', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--top-n', type=int, default=50, help='Number of top words to analyze')
    parser.add_argument('--min-freq', type=int, default=10, help='Minimum word frequency')
    parser.add_argument('--max-contexts', type=int, default=200, help='Maximum contexts per word')
    parser.add_argument('--log', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log)
    
    # Process Edo corpus
    if args.edo_dir and os.path.exists(args.edo_dir):
        process_corpus(args.edo_dir, args.output, 'edo', args.top_n, args.min_freq, args.max_contexts)
    else:
        logger.warning(f"Edo directory not found or not specified: {args.edo_dir}")
    
    # Process Meiji corpus
    if args.meiji_dir and os.path.exists(args.meiji_dir):
        process_corpus(args.meiji_dir, args.output, 'meiji', args.top_n, args.min_freq, args.max_contexts)
    else:
        logger.warning(f"Meiji directory not found or not specified: {args.meiji_dir}")
    
    logger.info("Preprocessing complete!")


if __name__ == '__main__':
    main()
