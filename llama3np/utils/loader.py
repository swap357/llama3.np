"""
Utilities for loading model weights and parameters.
"""

import numpy as np


def load_parameters(model_path: str) -> dict:
    """
    Load parameters from a model file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary of model parameters
    """
    return np.load(model_path)