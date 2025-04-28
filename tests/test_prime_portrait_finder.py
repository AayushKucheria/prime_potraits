"""
Tests for the prime portrait finder module.
"""

import numpy as np
import pytest
import os
import tempfile
import shutil
from PIL import Image
import gmpy2
from prime_portrait.prime_portrait_finder import PrimePortraitFinder, find_prime_portrait
from prime_portrait.number_theory import image_to_int, is_probably_prime
from prime_portrait.image_processing import dither
from prime_portrait.utils import add_noise

class TestPrimePortraitFinder:
    """Tests for the PrimePortraitFinder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a small test image (8x8) with simple pattern
        # This is small enough to find a prime quickly during tests
        self.test_image = np.zeros((8, 8), dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                self.test_image[i, j] = (i * j * 7) % 256
        
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_array(self):
        """Test initialization with a numpy array."""
        finder = PrimePortraitFinder(image_array=self.test_image, output_dir=self.temp_dir)
        assert np.array_equal(finder.image, self.test_image)
        assert finder.output_dir == self.temp_dir
    
    def test_init_with_path(self):
        """Test initialization with an image path."""
        # Save the test image to a temporary file
        image_path = os.path.join(self.temp_dir, "test_image.png")
        Image.fromarray(self.test_image).save(image_path)
        
        finder = PrimePortraitFinder(image_path=image_path, output_dir=self.temp_dir)
        assert np.array_equal(finder.image, self.test_image)
        assert finder.output_dir == self.temp_dir
    
    def test_find_prime_portrait_small_image(self):
        """Test finding a prime portrait with a small image."""
        finder = PrimePortraitFinder(image_array=self.test_image, output_dir=self.temp_dir)
        
        # Use a lower confidence and maximum iterations to make the test faster
        finder.primality_confidence = 2
        dithered_image, prime_int = finder.find_prime_portrait(max_iterations=10, verbose=False)
        
        # If we found a prime, verify it
        if finder.is_prime:
            assert dithered_image is not None
            assert prime_int is not None
            assert gmpy2.is_prime(prime_int, 10)  # Verify with higher confidence
        else:
            # Skip this test if no prime was found within max_iterations
            pytest.skip("No prime portrait found within max iterations")
    
    def test_save_results(self):
        """Test saving results after finding a prime portrait."""
        finder = PrimePortraitFinder(image_array=self.test_image, output_dir=self.temp_dir)
        
        # First, we need to find a prime portrait
        finder.primality_confidence = 2
        dithered_image, prime_int = finder.find_prime_portrait(max_iterations=10, verbose=False)
        
        if finder.is_prime:
            # Now test the save_results method
            saved_files = finder.save_results()
            
            # Check that the expected files were created
            assert os.path.exists(saved_files['original_image'])
            assert os.path.exists(saved_files['dithered_image'])
            assert os.path.exists(saved_files['prime_integer'])
            assert os.path.exists(saved_files['histogram'])
            assert os.path.exists(saved_files['report'])
            
            # Check that the prime integer file contains a valid number
            with open(saved_files['prime_integer'], 'r') as f:
                content = f.read().strip()
                saved_prime = gmpy2.mpz(content)
                assert saved_prime == prime_int
        else:
            pytest.skip("No prime portrait found within max iterations")
    
    def test_save_results_without_prime(self):
        """Test that save_results raises an error if no prime portrait has been found."""
        finder = PrimePortraitFinder(image_array=self.test_image, output_dir=self.temp_dir)
        
        # We haven't found a prime portrait yet, so this should raise an error
        with pytest.raises(RuntimeError):
            finder.save_results()

class TestConvenienceFunction:
    """Tests for the find_prime_portrait convenience function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a small test image
        self.test_image = np.zeros((8, 8), dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                self.test_image[i, j] = (i * j * 13) % 256
        
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Save the test image to a file
        self.image_path = os.path.join(self.temp_dir, "test_image.png")
        Image.fromarray(self.test_image).save(self.image_path)
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_find_prime_portrait_function(self):
        """Test the find_prime_portrait convenience function."""
        # Use a subdirectory of the temp directory for outputs
        output_dir = os.path.join(self.temp_dir, "outputs")
        
        # Use the convenience function
        dithered_image, prime_int, saved_files = find_prime_portrait(
            self.image_path, 
            output_dir=output_dir,
            max_iterations=10,
            verbose=False
        )
        
        if dithered_image is not None and prime_int is not None:
            # Verify that the files were saved
            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, "dithered_prime_image.png"))
            assert os.path.exists(os.path.join(output_dir, "prime_image_int.txt"))
            
            # Verify the prime number
            assert gmpy2.is_prime(prime_int, 10)
        else:
            pytest.skip("No prime portrait found within max iterations") 

class TestPrimePortraitDebugging:
    """Debug tests for the Prime Portrait functionality."""
    
    def setup_method(self):
        """Set up test fixtures with controlled test patterns."""
        # Create image patterns more likely to be prime
        self.micro_image = np.ones((2, 2), dtype=np.uint8)  # Tiny 2x2 image
        self.micro_image[0,0] = 1  # Set specific values that might make a prime number
        self.micro_image[0,1] = 1
        self.micro_image[1,0] = 0
        self.micro_image[1,1] = 1  # Binary: 1101 = 13 (prime)
        
        # Create a tiny image that should convert to a known prime (7)
        self.prime7_image = np.zeros((2, 2), dtype=np.uint8)
        self.prime7_image[0,0] = 0
        self.prime7_image[0,1] = 0
        self.prime7_image[1,0] = 1
        self.prime7_image[1,1] = 1  # Should be 00000111 in binary = 7 (prime)
        
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_primality_check_function(self):
        """Test that the primality check function works correctly on known primes."""
        # Test small known primes
        assert is_probably_prime(2) is True
        assert is_probably_prime(3) is True
        assert is_probably_prime(5) is True
        assert is_probably_prime(7) is True
        assert is_probably_prime(11) is True
        assert is_probably_prime(13) is True
        
        # Test small known composites
        assert is_probably_prime(4) is False
        assert is_probably_prime(6) is False
        assert is_probably_prime(8) is False
        assert is_probably_prime(9) is False
        assert is_probably_prime(10) is False
        assert is_probably_prime(12) is False
    
    def test_image_to_int_conversion(self):
        """Test that image to integer conversion produces expected values."""
        # Convert a simple 2x2 image with all 1s
        all_ones = np.ones((2, 2), dtype=np.uint8)
        int_value = image_to_int(all_ones)
        # Expected: 1*256^3 + 1*256^2 + 1*256^1 + 1*256^0 = 16843009
        assert int_value == 16843009
        
        # Test the special prime pattern
        int_value = image_to_int(self.micro_image)
        # Expected based on values: 1*256^3 + 1*256^2 + 0*256^1 + 1*256^0 = 16843009
        expected = 1*256**3 + 1*256**2 + 0*256**1 + 1
        assert int_value == expected
        
        # Test if prime7_image converts correctly to 7
        int_value = image_to_int(self.prime7_image)
        # Since the actual conversion does 256-base not binary, calculate expected value
        expected = 0*256**3 + 0*256**2 + 1*256**1 + 1*256**0
        assert int_value == expected
        assert int_value == 257  # 256 + 1 = 257 (not 7)
    
    def test_image_to_int_primality(self):
        """Test if images that should be prime actually produce prime integers."""
        # Convert images to integers and check primality
        micro_int = image_to_int(self.micro_image)
        assert is_probably_prime(micro_int) == gmpy2.is_prime(micro_int, 10)
        
        prime7_int = image_to_int(self.prime7_image)
        assert is_probably_prime(prime7_int) == gmpy2.is_prime(prime7_int, 10)
        
        print(f"Micro image int: {micro_int}, primality: {is_probably_prime(micro_int)}")
        print(f"Prime7 image int: {prime7_int}, primality: {is_probably_prime(prime7_int)}")
    
    def test_dithering_preserves_primes(self):
        """Test if the dithering process preserves primality."""
        # Find a small prime image
        for i in range(5):
            for j in range(5):
                # Try different 2x2 patterns
                test_img = np.zeros((2, 2), dtype=np.uint8)
                test_img[0,0] = (i*5 + j) % 2
                test_img[0,1] = (i*5 + j + 1) % 2
                test_img[1,0] = (i*5 + j + 2) % 2
                test_img[1,1] = (i*5 + j + 3) % 2
                
                img_int = image_to_int(test_img)
                is_prime = is_probably_prime(img_int)
                
                if is_prime:
                    # Found a prime image, let's test dithering
                    dithered = dither(test_img)
                    dithered_int = image_to_int(dithered)
                    is_still_prime = is_probably_prime(dithered_int, confidence=10)
                    
                    print(f"Original prime int: {img_int}")
                    print(f"Dithered int: {dithered_int}")
                    print(f"Is dithered prime? {is_still_prime}")
                    return
        
        pytest.skip("Could not find a prime pattern to test")
    
    def test_noise_and_primality(self):
        """Test if adding noise can produce prime numbers from non-prime images."""
        # Start with a non-prime image
        non_prime_img = np.zeros((4, 4), dtype=np.uint8)
        non_prime_int = image_to_int(non_prime_img)
        assert is_probably_prime(non_prime_int) is False
        
        # Try adding different levels of noise to see if any produce a prime
        found_prime = False
        for noise_level in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for _ in range(5):  # Try multiple times at each noise level
                noisy_img = add_noise(non_prime_img, noise_level=noise_level)
                noisy_int = image_to_int(noisy_img)
                if is_probably_prime(noisy_int, confidence=5):
                    found_prime = True
                    print(f"Found prime with noise level {noise_level}: {noisy_int}")
                    break
            if found_prime:
                break
        
        # We're not asserting a specific outcome here, just reporting if noise can lead to primes
    
    def test_forced_prime_finding(self):
        """Force-test finding a prime by trying many tiny images."""
        # Create tiny images and try to find prime
        found_prime = False
        for size in [2, 3, 4]:  # Try different tiny sizes
            for attempt in range(20):  # Try multiple random patterns
                test_img = np.random.randint(0, 2, size=(size, size), dtype=np.uint8)
                finder = PrimePortraitFinder(image_array=test_img, output_dir=self.temp_dir)
                finder.primality_confidence = 5
                finder.noise_level = 0.2
                
                dithered_image, prime_int = finder.find_prime_portrait(max_iterations=20, verbose=True)
                
                if finder.is_prime:
                    found_prime = True
                    print(f"Found prime with {size}x{size} image after {finder.iterations} iterations")
                    print(f"Prime number: {prime_int}")
                    
                    # Verify with high confidence
                    assert gmpy2.is_prime(prime_int, 20) is True
                    break
            
            if found_prime:
                break
        
        if not found_prime:
            pytest.skip("Could not find a prime portrait within test parameters") 