"""
Utility functions for the Edo-Meiji polysemy analysis project.

This module provides common utilities including:
- Logging configuration
- Device management (CPU/CUDA)
- File I/O helpers
- Progress tracking
"""

import logging
import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Optional path to log file. If None, logs to console only.
        level: Logging level (default: logging.INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("edo_meiji_analysis")
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device(use_cuda: bool = True) -> 'torch.device':
    """
    Get the appropriate PyTorch device (CPU or CUDA).
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        torch.device object
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this function. Install with: pip install torch")
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger = logging.getLogger("edo_meiji_analysis")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger = logging.getLogger("edo_meiji_analysis")
        logger.info("Using CPU device")
    
    return device


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Object to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def count_files(directory: str, extension: str = ".txt") -> int:
    """
    Count files with a given extension in a directory.
    
    Args:
        directory: Directory path
        extension: File extension to count (default: .txt)
        
    Returns:
        Number of matching files
    """
    path = Path(directory)
    return len(list(path.glob(f"*{extension}")))


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2h 34m 12s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
