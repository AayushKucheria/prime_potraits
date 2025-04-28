"""
Prime Portrait Finder module - core functionality for finding prime number images.
"""

import numpy as np
import os
import time
import gmpy2
from PIL import Image
import matplotlib.pyplot as plt
from .image_processing import dither, visualize_histograms
from .number_theory import image_to_int, is_probably_prime, save_int_to_file
from .utils import Timer, profile_memory, add_noise

class PrimePortraitFinder:
    """
    Core class to find prime portraits from images by iteratively adding noise
    and dithering until a prime number representation is found.
    """
    
    def __init__(self, image_path=None, image_array=None, output_dir="prime_portrait_output"):
        """
        Initialize the Prime Portrait Finder.
        
        Args:
            image_path: Path to the grayscale image file
            image_array: Numpy array of the grayscale image (alternative to image_path)
            output_dir: Directory to save output files
        """
        if image_path is not None:
            self.image = np.array(Image.open(image_path).convert('L'))
        elif image_array is not None:
            self.image = image_array.astype(np.uint8)
        else:
            raise ValueError("Either image_path or image_array must be provided")
            
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Parameters for prime portrait finding
        self.use_mp = True               # Use multiprocessing by default
        self.use_hybrid = True           # Use hybrid approach by default
        self.primality_confidence = 10   # Default primality test confidence
        self.noise_level = 0.01          # Default noise level (1%)
        self.early_terminate = True      # Use early termination optimization
        
        # Statistics
        self.total_time = 0
        self.iterations = 0
        self.noise_time = 0
        self.dither_time = 0
        self.conversion_time = 0
        self.primality_time = 0
        
        # Results
        self.result_image = None
        self.result_prime = None
        self.is_prime = False
        
        # Optimization
        self._bench_conversion_methods()
        
    def _bench_conversion_methods(self):
        """
        Benchmark integer conversion methods and use the fastest for this image size.
        """
        print("Benchmarking integer conversion methods...")
        
        # Test multiprocessing vs serial
        with Timer("MP conversion") as mp_timer:
            _ = image_to_int(self.image, use_mp=True, use_hybrid=True)
        
        with Timer("Serial conversion") as serial_timer:
            _ = image_to_int(self.image, use_mp=False, use_hybrid=True)
        
        # Choose the faster method
        self.use_mp = mp_timer.elapsed < serial_timer.elapsed
        print(f"Using {'multiprocessing' if self.use_mp else 'serial'} processing")
        
        # Test hybrid approach vs direct
        with Timer("Hybrid conversion") as hybrid_timer:
            _ = image_to_int(self.image, use_mp=self.use_mp, use_hybrid=True)
        
        with Timer("Direct conversion") as direct_timer:
            _ = image_to_int(self.image, use_mp=self.use_mp, use_hybrid=False)
        
        # Choose the faster approach
        self.use_hybrid = hybrid_timer.elapsed < direct_timer.elapsed
        print(f"Using {'hybrid' if self.use_hybrid else 'direct'} approach")
    
    def find_prime_portrait(self, max_iterations=None, verbose=True):
        """
        Find a prime portrait by iteratively adding noise and dithering.
        
        Args:
            max_iterations: Maximum number of iterations to try
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (prime_image, prime_integer)
        """
        self.is_prime = False
        self.iterations = 0
        current_image = self.image.copy().astype(float)
        
        # Initialize gmpy2 context for better performance with large integers
        orig_ctx = gmpy2.get_context().copy()
        ctx = gmpy2.get_context()
        ctx.precision = 10000  # Higher precision for large integers
        gmpy2.set_context(ctx)
        
        start_time = time.time()
        
        try:
            # Do-while loop until a prime number is found or max iterations reached
            while not self.is_prime:
                self.iterations += 1
                loop_start = time.time()
                
                if verbose and self.iterations % 10 == 0:
                    print(f"\nIteration {self.iterations}")
                    print(f"Memory usage: {profile_memory():.2f} MB")
                
                # Add random noise
                with Timer() as noise_timer:
                    noisy_image = add_noise(current_image.astype(np.uint8), noise_level=self.noise_level)
                self.noise_time += noise_timer.elapsed
                
                # Apply dithering
                with Timer() as dither_timer:
                    dithered_image = dither(noisy_image)
                self.dither_time += dither_timer.elapsed
                
                # Convert to integer and check primality
                with Timer() as conversion_timer:
                    image_int = image_to_int(dithered_image, use_mp=self.use_mp, use_hybrid=self.use_hybrid)
                self.conversion_time += conversion_timer.elapsed
                
                with Timer() as primality_timer:
                    # Fast primality testing strategy
                    if self.early_terminate and self.iterations % 10 != 0:
                        # Quick check with low confidence
                        self.is_prime = is_probably_prime(image_int, confidence=1)
                        if self.is_prime:
                            # Verify with higher confidence if it passes the fast test
                            self.is_prime = is_probably_prime(image_int, confidence=self.primality_confidence)
                    else:
                        # Full test with higher confidence
                        self.is_prime = is_probably_prime(image_int, confidence=self.primality_confidence)
                self.primality_time += primality_timer.elapsed
                
                if verbose and self.iterations % 10 == 0:
                    print(f"  Noise generation: {noise_timer.elapsed:.4f}s")
                    print(f"  Dithering: {dither_timer.elapsed:.4f}s")
                    print(f"  Int conversion: {conversion_timer.elapsed:.4f}s")
                    print(f"  Primality check: {primality_timer.elapsed:.4f}s")
                    print(f"  Is prime? {self.is_prime}")
                
                # Use the current dithered image as base for next iteration
                current_image = dithered_image
                
                # Check if we've reached the maximum iterations
                if max_iterations is not None and self.iterations >= max_iterations:
                    print(f"Reached maximum iterations ({max_iterations}) without finding a prime")
                    break
                
                loop_time = time.time() - loop_start
                self.total_time += loop_time
        
        finally:
            # Restore original context
            gmpy2.set_context(orig_ctx)
        
        if self.is_prime:
            self.result_image = dithered_image
            self.result_prime = image_int
            
            if verbose:
                print(f"\nSuccess! Found a prime number after {self.iterations} iterations.")
                print(f"Total time: {self.total_time:.2f} seconds")
                
            return dithered_image, image_int
        else:
            return None, None
    
    def save_results(self):
        """
        Save all results from the prime portrait finding process.
        
        Returns:
            Dictionary of saved file paths
        """
        if not self.is_prime or self.result_image is None:
            raise RuntimeError("No prime portrait has been found yet. Call find_prime_portrait first.")
        
        saved_files = {}
        
        # Save the original image
        original_path = os.path.join(self.output_dir, 'original_image.png')
        Image.fromarray(self.image).save(original_path)
        saved_files['original_image'] = original_path
        
        # Save the dithered prime image
        dithered_path = os.path.join(self.output_dir, 'dithered_prime_image.png')
        Image.fromarray(self.result_image).save(dithered_path)
        saved_files['dithered_image'] = dithered_path
        
        # Save the prime integer
        prime_path = os.path.join(self.output_dir, 'prime_image_int.txt')
        save_int_to_file(self.result_prime, prime_path)
        saved_files['prime_integer'] = prime_path
        
        # Create and save histograms
        hist_path = os.path.join(self.output_dir, 'histogram.png')
        visualize_histograms(self.image, self.result_image, hist_path)
        saved_files['histogram'] = hist_path
        
        # Save a report with statistics
        report_path = os.path.join(self.output_dir, 'report.txt')
        with open(report_path, 'w') as f:
            f.write("===== PRIME PORTRAIT REPORT =====\n\n")
            f.write(f"Total iterations: {self.iterations}\n")
            f.write(f"Total time: {self.total_time:.2f}s\n")
            f.write(f"Average time per iteration: {self.total_time/self.iterations:.4f}s\n\n")
            f.write(f"Time breakdown:\n")
            f.write(f"  Noise generation: {self.noise_time:.2f}s ({self.noise_time/self.total_time*100:.1f}%)\n")
            f.write(f"  Dithering: {self.dither_time:.2f}s ({self.dither_time/self.total_time*100:.1f}%)\n")
            f.write(f"  Int conversion: {self.conversion_time:.2f}s ({self.conversion_time/self.total_time*100:.1f}%)\n")
            f.write(f"  Primality check: {self.primality_time:.2f}s ({self.primality_time/self.total_time*100:.1f}%)\n\n")
            
            # Add information about the final prime number
            f.write(f"Prime number found:\n")
            f.write(f"  Decimal digits: {len(str(self.result_prime))}\n")
            f.write(f"  Binary digits: {self.result_prime.bit_length()}\n")
            
            # Add primality verification info
            final_check = gmpy2.is_prime(self.result_prime, 50)
            f.write(f"  Final primality check with maximum confidence: {final_check}\n")
        
        saved_files['report'] = report_path
        
        return saved_files

def find_prime_portrait(image_path, output_dir="prime_portrait_output", max_iterations=None, verbose=True):
    """
    Convenience function to find a prime portrait from an image file.
    
    Args:
        image_path: Path to the grayscale image file
        output_dir: Directory to save output files
        max_iterations: Maximum number of iterations to try
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (dithered_image, prime_integer, saved_files)
    """
    finder = PrimePortraitFinder(image_path=image_path, output_dir=output_dir)
    dithered_image, prime_integer = finder.find_prime_portrait(max_iterations=max_iterations, verbose=verbose)
    
    if dithered_image is not None and prime_integer is not None:
        saved_files = finder.save_results()
        return dithered_image, prime_integer, saved_files
    
    return None, None, None 