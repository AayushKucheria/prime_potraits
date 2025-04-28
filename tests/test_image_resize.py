"""
Tests for the image resizing functionality.
"""

import numpy as np
import pytest
import os
import tempfile
from PIL import Image
from prime_portrait.image_processing import resize_image

class TestImageResize:
    """Tests for the image resizing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a test image (100x100) with a simple pattern
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            for j in range(100):
                self.test_image[i, j] = (i + j) % 256
        
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Save the test image to a file
        self.image_path = os.path.join(self.temp_dir, "test_image.png")
        Image.fromarray(self.test_image).save(self.image_path)
    
    def teardown_method(self):
        """Tear down test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_resize_with_max_size(self):
        """Test resizing with a maximum size constraint."""
        # Original size: 100x100
        # Max size: 50, should resize to 50x50
        resized = resize_image(self.test_image, max_size=50)
        assert resized.shape == (50, 50)
        
        # Check that pixel values are still in valid range
        assert resized.min() >= 0
        assert resized.max() <= 255
    
    def test_resize_with_target_pixels(self):
        """Test resizing to target number of pixels."""
        # Original size: 100x100 = 10,000 pixels
        # Target: 2500 pixels, should be approximately 50x50
        resized = resize_image(self.test_image, target_pixels=2500)
        # Allow a small margin of error due to aspect ratio preservation
        pixel_count = resized.shape[0] * resized.shape[1]
        assert abs(pixel_count - 2500) < 100
    
    def test_resize_does_not_affect_small_images(self):
        """Test that images smaller than the constraints are not resized."""
        small_image = np.zeros((30, 30), dtype=np.uint8)
        
        # Max size is 50, should not resize
        resized = resize_image(small_image, max_size=50)
        assert resized.shape == (30, 30)
        
        # Target 2500 pixels, should not resize
        resized = resize_image(small_image, target_pixels=2500)
        assert resized.shape == (30, 30)
    
    def test_resize_from_file_path(self):
        """Test resizing from a file path."""
        resized = resize_image(image_path=self.image_path, max_size=50)
        assert resized.shape == (50, 50)
    
    def test_resize_invalid_inputs(self):
        """Test that resize_image raises appropriate errors for invalid inputs."""
        # No image or path provided
        with pytest.raises(ValueError):
            resize_image()
        
        # Both image and path provided
        with pytest.raises(ValueError):
            resize_image(self.test_image, image_path=self.image_path)
        
        # Invalid max_size
        with pytest.raises(ValueError):
            resize_image(self.test_image, max_size=-10)
        
        # Invalid target_pixels
        with pytest.raises(ValueError):
            resize_image(self.test_image, target_pixels=-2500)
    
    def test_resize_quality_preservation(self):
        """Test that resizing preserves image patterns."""
        resized = resize_image(self.test_image, max_size=50)
        
        # Calculate average pixel value for both images
        orig_avg = np.mean(self.test_image)
        resized_avg = np.mean(resized)
        
        # The average value should be approximately preserved
        assert abs(orig_avg - resized_avg) < 5 