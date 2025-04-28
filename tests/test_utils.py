"""
Tests for the utils module.
"""

import numpy as np
import pytest
import time
import io
import sys
from prime_portrait.utils import profile_memory, add_noise, Timer

class TestMemoryProfiling:
    """Tests for the memory profiling functions."""
    
    def test_profile_memory_returns_positive(self):
        """Test that profile_memory returns a positive value."""
        memory = profile_memory()
        assert memory > 0
    
    def test_profile_memory_returns_float(self):
        """Test that profile_memory returns a float."""
        memory = profile_memory()
        assert isinstance(memory, float)

class TestImageNoise:
    """Tests for the image noise functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_image = np.zeros((50, 50), dtype=np.uint8)
        # Set all pixels to 128 (middle gray)
        self.test_image.fill(128)
    
    def test_add_noise_shape(self):
        """Test that add_noise preserves the image shape."""
        noisy = add_noise(self.test_image)
        assert noisy.shape == self.test_image.shape
    
    def test_add_noise_type(self):
        """Test that add_noise returns a uint8 array."""
        noisy = add_noise(self.test_image)
        assert noisy.dtype == np.uint8
    
    def test_add_noise_level(self):
        """Test that add_noise adds the expected amount of noise."""
        noisy = add_noise(self.test_image, noise_level=0.01)
        # Since we set all pixels to 128, the difference is the noise
        diff = noisy.astype(np.float32) - self.test_image
        # Check that at least some pixels changed
        assert np.any(diff != 0)
        # Check that the noise is within the expected range
        max_expected_diff = 0.01 * 255 + 1  # 1% of 255 plus potential rounding
        assert np.max(np.abs(diff)) <= max_expected_diff
    
    def test_add_noise_different_each_time(self):
        """Test that add_noise produces different results each time."""
        noisy1 = add_noise(self.test_image)
        noisy2 = add_noise(self.test_image)
        # The two noisy images should be different
        assert not np.array_equal(noisy1, noisy2)
    
    def test_add_noise_respects_bounds(self):
        """Test that add_noise does not produce values outside the valid range."""
        # Create an image with values near the bounds
        edge_image = np.zeros((50, 50), dtype=np.uint8)
        edge_image[:25] = 0    # First half: black
        edge_image[25:] = 255  # Second half: white
        
        noisy = add_noise(edge_image, noise_level=0.1)
        
        # Check that the values are still within bounds
        assert np.min(noisy) >= 0
        assert np.max(noisy) <= 255

class TestTimer:
    """Tests for the Timer class."""
    
    def test_timer_measures_time(self):
        """Test that the Timer measures elapsed time."""
        with Timer() as timer:
            time.sleep(0.1)  # Sleep for 0.1 seconds
        
        # Check that the timer measured approximately 0.1 seconds
        assert 0.05 < timer.elapsed < 0.15
    
    def test_timer_with_name_prints_output(self):
        """Test that the Timer prints output when a name is provided."""
        # Redirect stdout to capture the output
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            with Timer(name="Test Timer"):
                pass
            
            output = sys.stdout.getvalue()
            assert "Test Timer:" in output
            assert "seconds" in output
        finally:
            # Restore stdout
            sys.stdout = stdout 