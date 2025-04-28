"""
Tests for advanced image resizing functionality.
"""

import numpy as np
import pytest
from PIL import Image
import os
import tempfile
import math
from prime_portrait.image_processing import resize_image, resize_for_prime_portrait


class TestAdvancedResize:
    """Tests for advanced image resizing functionality."""
    
    def setup_method(self):
        """Set up test images of different shapes and aspect ratios."""
        # Create test images
        self.square_image = np.zeros((100, 100), dtype=np.uint8)
        self.wide_image = np.zeros((100, 200), dtype=np.uint8)
        self.tall_image = np.zeros((200, 100), dtype=np.uint8)
        self.large_image = np.zeros((1000, 1000), dtype=np.uint8)
        
        # Add some gradient patterns to make the images visually meaningful
        for i in range(self.square_image.shape[0]):
            for j in range(self.square_image.shape[1]):
                self.square_image[i, j] = (i + j) % 256
                
        for i in range(self.wide_image.shape[0]):
            for j in range(self.wide_image.shape[1]):
                self.wide_image[i, j] = (i * 2 + j) % 256
                
        for i in range(self.tall_image.shape[0]):
            for j in range(self.tall_image.shape[1]):
                self.tall_image[i, j] = (i + j * 2) % 256
                
        for i in range(0, self.large_image.shape[0], 10):
            for j in range(0, self.large_image.shape[1], 10):
                # Create a checkboard pattern
                value = 255 if ((i // 10) + (j // 10)) % 2 == 0 else 0
                self.large_image[i:i+10, j:j+10] = value
    
    def test_existing_resize_function(self):
        """Test the existing resize_image function."""
        # Make sure the existing function works as expected
        resized = resize_image(self.square_image, max_size=50)
        assert resized.shape[0] <= 50
        assert resized.shape[1] <= 50
        
        # Test maintaining aspect ratio
        resized_wide = resize_image(self.wide_image, max_size=50)
        aspect_ratio_original = self.wide_image.shape[1] / self.wide_image.shape[0]
        aspect_ratio_resized = resized_wide.shape[1] / resized_wide.shape[0]
        assert abs(aspect_ratio_original - aspect_ratio_resized) < 0.1
    
    def test_resize_for_prime_portrait(self):
        """Test the new resize_for_prime_portrait function."""
        # Test with different target pixel counts
        for target_pixels in [100, 500, 1000]:
            # Test square image
            resized = resize_for_prime_portrait(self.square_image, target_pixels=target_pixels)
            actual_pixels = resized.shape[0] * resized.shape[1]
            # Should be within 10% of target
            assert actual_pixels >= target_pixels * 0.9
            assert actual_pixels <= target_pixels * 1.1
            
            # Test wide image
            resized = resize_for_prime_portrait(self.wide_image, target_pixels=target_pixels)
            actual_pixels = resized.shape[0] * resized.shape[1]
            assert actual_pixels >= target_pixels * 0.9
            assert actual_pixels <= target_pixels * 1.1
            
            # Test tall image
            resized = resize_for_prime_portrait(self.tall_image, target_pixels=target_pixels)
            actual_pixels = resized.shape[0] * resized.shape[1]
            assert actual_pixels >= target_pixels * 0.9
            assert actual_pixels <= target_pixels * 1.1
            
            # Verify aspect ratio is preserved
            aspect_ratio_original = self.tall_image.shape[1] / self.tall_image.shape[0]
            aspect_ratio_resized = resized.shape[1] / resized.shape[0]
            assert abs(aspect_ratio_original - aspect_ratio_resized) < 0.1
    
    def test_two_phase_resize(self):
        """Test the two-phase resize strategy."""
        # Only test with large image to see if two-phase approach helps quality
        target_pixels = 500
        
        # Two-phase approach
        resized = resize_for_prime_portrait(self.large_image, target_pixels=target_pixels, use_two_phase=True)
        actual_pixels = resized.shape[0] * resized.shape[1]
        
        # Check pixel count
        assert actual_pixels >= target_pixels * 0.9
        assert actual_pixels <= target_pixels * 1.1
        
        # Single-phase approach (for comparison)
        resized_single = resize_for_prime_portrait(self.large_image, target_pixels=target_pixels, use_two_phase=False)
        
        # Both should have similar pixel counts
        assert abs(resized.size - resized_single.size) < target_pixels * 0.1
    
    def test_very_small_targets(self):
        """Test with very small target pixel counts."""
        # Test with an extremely small target
        target_pixels = 25  # 5x5 equivalent
        
        resized = resize_for_prime_portrait(self.large_image, target_pixels=target_pixels)
        actual_pixels = resized.shape[0] * resized.shape[1]
        
        # Even for tiny targets, should be reasonably close
        assert actual_pixels >= target_pixels * 0.8
        assert actual_pixels <= target_pixels * 1.2
    
    def test_max_dimension_constraint(self):
        """Test that a maximum dimension constraint works with target pixels."""
        target_pixels = 1000
        max_dimension = 20
        
        resized = resize_for_prime_portrait(self.square_image, target_pixels=target_pixels, 
                                           max_dimension=max_dimension)
        
        # The max dimension should be respected
        assert resized.shape[0] <= max_dimension
        assert resized.shape[1] <= max_dimension
        
        # And the result should be as close as possible to the target pixels
        # while respecting the max dimension
        max_possible_pixels = max_dimension * max_dimension
        if target_pixels > max_possible_pixels:
            assert resized.shape[0] * resized.shape[1] <= max_possible_pixels
        else:
            assert resized.shape[0] * resized.shape[1] >= target_pixels * 0.9 