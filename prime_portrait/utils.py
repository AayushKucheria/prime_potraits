"""
Utility functions for the Prime Portrait package.
"""

import psutil
import time
import os
import random
import numpy as np

def profile_memory():
    """
    Returns current memory usage in MB.
    
    Returns:
        Float representing memory usage in MB
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def add_noise(image, noise_level=0.01):
    """
    Add random noise to an image.
    
    Args:
        image: A numpy array representing the grayscale image
        noise_level: The noise level to add (0.01 = 1%)
        
    Returns:
        A numpy array representing the noisy image
    """
    noise = np.random.uniform(-noise_level, noise_level, image.shape) * 255
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy

class Timer:
    """Simple timer for profiling code performance."""
    
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f} seconds") 