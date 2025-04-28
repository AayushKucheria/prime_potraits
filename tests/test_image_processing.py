"""
Tests for the image processing module.
"""

import numpy as np
import pytest
from PIL import Image
import matplotlib.pyplot as plt
import os
import tempfile
from prime_portrait.image_processing import dither, visualize_histograms

class TestDithering:
    """Tests for the dithering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test image with a gradient
        size = 50
        self.test_image = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            self.test_image[:, i] = int(i * (255 / size))
    
    def test_dither_output_shape(self):
        """Test that the dither function preserves the image shape."""
        dithered = dither(self.test_image)
        assert dithered.shape == self.test_image.shape
    
    def test_dither_output_type(self):
        """Test that the dither function returns a uint8 array."""
        dithered = dither(self.test_image)
        assert dithered.dtype == np.uint8
    
    def test_dither_reduces_unique_values(self):
        """Test that dithering reduces the number of unique pixel values."""
        # The original gradient should have many unique values
        original_unique = len(np.unique(self.test_image))
        
        # Dithered image should have fewer unique values due to quantization
        dithered = dither(self.test_image)
        dithered_unique = len(np.unique(dithered))
        
        # Our dithering uses 10 levels, so we expect 10 or fewer unique values
        assert dithered_unique <= 10
        assert dithered_unique < original_unique

    def test_dither_preserves_average_intensity(self):
        """Test that dithering approximately preserves the average intensity."""
        original_mean = np.mean(self.test_image)
        dithered = dither(self.test_image)
        dithered_mean = np.mean(dithered)
        # Print both means for debugging and verification
        print(f"Original image mean: {original_mean:.2f}")
        print(f"Dithered image mean: {dithered_mean:.2f}")
        # Allow for some variance due to dithering algorithm
        assert abs(original_mean - dithered_mean) < 10

class TestVisualization:
    """Tests for the visualization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test images
        self.original = np.zeros((50, 50), dtype=np.uint8)
        self.dithered = np.zeros((50, 50), dtype=np.uint8)
        
        # Add some varying values
        for i in range(50):
            self.original[:, i] = i * 5
            self.dithered[:, i] = (i // 5) * 25
    
    def test_visualize_histograms_returns_figure(self):
        """Test that visualize_histograms returns a matplotlib figure."""
        fig = visualize_histograms(self.original, self.dithered)
        assert isinstance(fig, plt.Figure)
    
    def test_visualize_histograms_saves_file(self):
        """Test that visualize_histograms saves a file when a path is provided."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "histogram.png")
            visualize_histograms(self.original, self.dithered, save_path)
            assert os.path.exists(save_path) 