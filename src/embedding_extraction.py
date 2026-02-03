"""
Embedding extraction module using Japanese BERT.

This module handles:
- Loading cl-tohoku/bert-base-japanese model
- Batch processing of sentences
- Extracting contextual embeddings for target words
- Saving embeddings for downstream analysis
"""

import argparse
import logging
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

try:
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

from utils import setup_logging, get_device, ensure_dir, save_pickle, load_json


class BERTEmbeddingExtractor:
    """
    Extract contextual embeddings using Japanese BERT.
    """
    
    def __init__(self, model_name: str = 'cl-tohoku/bert-base-japanese', device: str = 'auto'):
        """
        Initialize BERT model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('auto', 'cuda', or 'cpu')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")
        
        self.logger = logging.getLogger("edo_meiji_analysis")
        self.logger.info(f"Loading BERT model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
        # Set device
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("BERT model loaded successfully")
    
    def extract_word_embedding(
        self,
        sentence: str,
        target_word: str,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        Extract embedding for a target word in a sentence.
        
        Args:
            sentence: Input sentence
            target_word: Word to extract embedding for
            aggregation: How to aggregate subword embeddings ('mean', 'first', 'last')
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # Find target word tokens
        tokens = self.tokenizer.tokenize(sentence)
        word_token_ids = self._find_word_positions(tokens, target_word)
        
        if not word_token_ids:
            # If word not found, return CLS token embedding
            self.logger.warning(f"Target word '{target_word}' not found in sentence, using CLS")
            return last_hidden_state[0, 0, :].cpu().numpy()
        
        # Extract embeddings for target word tokens (add 1 to account for CLS token)
        word_embeddings = last_hidden_state[0, [idx + 1 for idx in word_token_ids], :]
        
        # Aggregate subword embeddings
        if aggregation == 'mean':
            embedding = torch.mean(word_embeddings, dim=0)
        elif aggregation == 'first':
            embedding = word_embeddings[0]
        elif aggregation == 'last':
            embedding = word_embeddings[-1]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return embedding.cpu().numpy()
    
    def _find_word_positions(self, tokens: List[str], target_word: str) -> List[int]:
        """
        Find the positions of target word tokens in the tokenized sequence.
        
        Args:
            tokens: List of tokens
            target_word: Target word to find
            
        Returns:
            List of token indices
        """
        positions = []
        word_length = len(target_word)
        
        # Try exact match first
        for i, token in enumerate(tokens):
            if token == target_word:
                positions.append(i)
        
        if positions:
            return positions
        
        # Try substring match (for subword tokenization)
        token_str = ''.join(tokens).replace('##', '')
        if target_word in token_str:
            # Find approximate positions
            i = 0
            current_pos = 0
            for token in tokens:
                clean_token = token.replace('##', '')
                if target_word[0] in clean_token:
                    positions.append(i)
                i += 1
                current_pos += len(clean_token)
                if current_pos >= word_length:
                    break
        
        return positions
    
    def extract_embeddings_for_word(
        self,
        contexts: List[str],
        target_word: str,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract embeddings for a word across multiple contexts.
        
        Args:
            contexts: List of sentences containing the target word
            target_word: Word to extract embeddings for
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_contexts, embedding_dim)
        """
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(contexts), batch_size), 
                     desc=f"Extracting embeddings for '{target_word}'",
                     leave=False):
            batch_contexts = contexts[i:i + batch_size]
            
            for context in batch_contexts:
                embedding = self.extract_word_embedding(context, target_word)
                embeddings.append(embedding)
        
        return np.array(embeddings)


def extract_embeddings_for_corpus(
    contexts_file: str,
    output_dir: str,
    model_name: str = 'cl-tohoku/bert-base-japanese',
    batch_size: int = 32,
    device: str = 'auto'
) -> None:
    """
    Extract embeddings for all words in a corpus.
    
    Args:
        contexts_file: Path to contexts JSON file
        output_dir: Output directory for embeddings
        model_name: BERT model identifier
        batch_size: Batch size for processing
        device: Device to use
    """
    logger = logging.getLogger("edo_meiji_analysis")
    logger.info(f"Processing contexts from {contexts_file}")
    
    # Load contexts
    contexts_data = load_json(contexts_file)
    era = contexts_data['era']
    words_data = contexts_data['words']
    
    logger.info(f"Found {len(words_data)} words in {era} corpus")
    
    # Initialize extractor
    extractor = BERTEmbeddingExtractor(model_name, device)
    
    # Extract embeddings for each word
    embeddings_data = {}
    
    for word, data in tqdm(words_data.items(), desc=f"Processing {era} words"):
        contexts = data['contexts']
        
        if not contexts:
            logger.warning(f"No contexts found for word '{word}', skipping")
            continue
        
        # Extract embeddings
        embeddings = extractor.extract_embeddings_for_word(contexts, word, batch_size)
        
        embeddings_data[word] = {
            'embeddings': embeddings,
            'contexts': contexts,
            'n_contexts': len(contexts)
        }
        
        logger.info(f"Extracted {len(embeddings)} embeddings for '{word}'")
    
    # Save embeddings
    ensure_dir(output_dir)
    output_file = os.path.join(output_dir, f'{era}_embeddings.pkl')
    save_pickle(embeddings_data, output_file)
    logger.info(f"Saved embeddings to {output_file}")


def main():
    """Main entry point for embedding extraction."""
    parser = argparse.ArgumentParser(description="Extract BERT embeddings for Japanese text")
    parser.add_argument('--input', type=str, required=True, help='Input directory with contexts')
    parser.add_argument('--output', type=str, default='data/embeddings', help='Output directory')
    parser.add_argument('--model', type=str, default='cl-tohoku/bert-base-japanese', 
                       help='BERT model name')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--log', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log)
    
    # Process both eras
    for era in ['edo', 'meiji']:
        contexts_file = os.path.join(args.input, f'{era}_contexts.json')
        
        if os.path.exists(contexts_file):
            extract_embeddings_for_corpus(
                contexts_file,
                args.output,
                args.model,
                args.batch_size,
                args.device
            )
        else:
            logger.warning(f"Contexts file not found: {contexts_file}")
    
    logger.info("Embedding extraction complete!")


if __name__ == '__main__':
    main()
