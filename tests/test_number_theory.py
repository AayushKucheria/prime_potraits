"""
Tests for the number theory module.
"""

import numpy as np
import pytest
import gmpy2
import tempfile
import os
from prime_portrait.number_theory import image_to_int, is_probably_prime, hash_image, save_int_to_file

class TestImageToInt:
    """Tests for the image_to_int function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.small_image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        
        # Create a larger test image
        self.large_image = np.zeros((32, 32), dtype=np.uint8)
        for i in range(32):
            for j in range(32):
                self.large_image[i, j] = (i * j) % 256
    
    def test_small_image_conversion(self):
        """Test conversion of a small, known image."""
        # For the image [[1, 2], [3, 4]], the flattened array is [1, 2, 3, 4]
        # Converting to a number in base 256: 1*256^3 + 2*256^2 + 3*256^1 + 4*256^0
        expected = 1*256**3 + 2*256**2 + 3*256**1 + 4
        result = image_to_int(self.small_image, use_mp=False, use_hybrid=True)
        assert result == expected
    
    def test_methods_produce_same_result(self):
        """Test that different conversion methods produce the same result."""
        # Direct method
        direct = image_to_int(self.small_image, use_mp=False, use_hybrid=False)
        
        # Hybrid method
        hybrid = image_to_int(self.small_image, use_mp=False, use_hybrid=True)
        
        assert direct == hybrid
    
    def test_large_image_conversion(self):
        """Test conversion of a larger image."""
        # Different methods should produce the same result
        direct = image_to_int(self.large_image, use_mp=False, use_hybrid=False)
        hybrid = image_to_int(self.large_image, use_mp=False, use_hybrid=True)
        parallel = image_to_int(self.large_image, use_mp=True, use_hybrid=True)
        
        assert direct == hybrid
        assert direct == parallel
    
    def test_zero_image_conversion(self):
        """Test conversion of an image with all zeros."""
        zero_image = np.zeros((5, 5), dtype=np.uint8)
        result = image_to_int(zero_image)
        assert result == 0

class TestPrimalityTesting:
    """Tests for the is_probably_prime function."""
    
    def test_known_primes(self):
        """Test known prime numbers."""
        known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for prime in known_primes:
            assert is_probably_prime(prime) is True
    
    def test_known_composites(self):
        """Test known composite numbers."""
        known_composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        for composite in known_composites:
            assert is_probably_prime(composite) is False
    
    def test_edge_cases(self):
        """Test edge cases."""
        assert is_probably_prime(0) is False
        assert is_probably_prime(1) is False
        assert is_probably_prime(2) is True
    
    def test_larger_prime(self):
        """Test a larger known prime number."""
        # A known prime number
        large_prime = gmpy2.mpz(101)
        assert is_probably_prime(large_prime) is True
        
    def test_larger_composite(self):
        """Test a larger known composite number."""
        # A known composite number
        large_composite = gmpy2.mpz(100)
        assert is_probably_prime(large_composite) is False
    
    def test_caching(self):
        """Test that caching works."""
        # First call should compute the result
        result1 = is_probably_prime(97)
        # Second call should use the cache
        result2 = is_probably_prime(97)
        assert result1 is True
        assert result2 is True

class TestHashAndSave:
    """Tests for the hash_image and save_int_to_file functions."""
    
    def test_hash_image_returns_string(self):
        """Test that hash_image returns a string."""
        image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = hash_image(image)
        assert isinstance(result, str)
    
    def test_hash_image_different_for_different_images(self):
        """Test that different images have different hashes."""
        image1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        image2 = np.array([[4, 3], [2, 1]], dtype=np.uint8)
        hash1 = hash_image(image1)
        hash2 = hash_image(image2)
        assert hash1 != hash2
    
    def test_save_int_to_file(self):
        """Test saving an integer to a file."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, "test_int.txt")
            
            # Test with a small number
            save_int_to_file(12345, file_path)
            with open(file_path, 'r') as f:
                content = f.read().strip()
            assert content == "12345"
            
            # Test with zero
            save_int_to_file(0, file_path)
            with open(file_path, 'r') as f:
                content = f.read().strip()
            assert content == "0"
            
            # Test with a large number
            large_num = gmpy2.mpz(10) ** 100 + 1  # 10^100 + 1
            save_int_to_file(large_num, file_path)
            with open(file_path, 'r') as f:
                content = f.read().strip()
            assert content == str(large_num) 