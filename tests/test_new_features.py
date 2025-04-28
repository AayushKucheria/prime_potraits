"""
Tests for new features we're developing using TDD.
"""

import numpy as np
import pytest
from prime_portrait.image_processing import adaptive_dither

class TestAdaptiveDithering:
    """Tests for the new adaptive dithering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test image with a gradient
        size = 50
        self.test_image = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            self.test_image[:, i] = int(i * (255 / size))
    
    def test_adaptive_dither_exists(self):
        """Test that the adaptive_dither function exists."""
        # This will fail until we implement the function
        assert callable(adaptive_dither)
    
    def test_adaptive_dither_output_shape(self):
        """Test that the adaptive_dither function preserves the image shape."""
        dithered = adaptive_dither(self.test_image)
        assert dithered.shape == self.test_image.shape
    
    def test_adaptive_dither_output_type(self):
        """Test that the adaptive_dither function returns a uint8 array."""
        dithered = adaptive_dither(self.test_image)
        assert dithered.dtype == np.uint8
    
    def test_adaptive_dither_detail_parameter(self):
        """Test that the detail parameter affects the output."""
        # Low detail should have fewer unique values
        low_detail = adaptive_dither(self.test_image, detail=0.2)
        # High detail should have more unique values
        high_detail = adaptive_dither(self.test_image, detail=0.8)
        
        low_unique = len(np.unique(low_detail))
        high_unique = len(np.unique(high_detail))
        
        # Higher detail should have more unique values or at least the same
        assert high_unique >= low_unique
    
    def test_adaptive_dither_preserves_average_intensity(self):
        """Test that adaptive dithering approximately preserves the average intensity."""
        original_mean = np.mean(self.test_image)
        dithered = adaptive_dither(self.test_image)
        dithered_mean = np.mean(dithered)
        
        # Allow for some variance due to dithering algorithm
        assert abs(original_mean - dithered_mean) < 10
    
    def test_adaptive_dither_different_patterns(self):
        """Test that different patterns produce different results."""
        # Test standard dithering pattern
        standard = adaptive_dither(self.test_image, pattern="standard")
        
        # Test ordered dithering pattern
        ordered = adaptive_dither(self.test_image, pattern="ordered")
        
        # Test blue noise dithering pattern
        blue_noise = adaptive_dither(self.test_image, pattern="blue_noise")
        
        # They should all be different
        assert not np.array_equal(standard, ordered)
        assert not np.array_equal(standard, blue_noise)
        assert not np.array_equal(ordered, blue_noise) 