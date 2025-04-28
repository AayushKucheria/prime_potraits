#!/usr/bin/env python3
"""
Debug script for prime portrait generation.
Uses a tiny image and adds extensive logging to diagnose issues.
"""

import numpy as np
from PIL import Image
import os
import tempfile
import gmpy2
from prime_portrait.prime_portrait_finder import PrimePortraitFinder
from prime_portrait.number_theory import image_to_int, is_probably_prime
from prime_portrait.image_processing import dither
from prime_portrait.utils import add_noise
import math

def main():
    # Create a very small test image with a higher chance of producing a prime
    print("Creating test image...")
    size = 3  # 3x3 is even smaller for faster prime finding
    test_image = np.zeros((size, size), dtype=np.uint8)
    
    # Manually set pixel values to increase likelihood of primality
    # Prime numbers often have certain patterns that might be exploitable
    for i in range(size):
        for j in range(size):
            # Alternating pattern that often leads to prime-friendly distributions
            test_image[i, j] = ((i + j) % 2) * 255
    
    print(f"Test image shape: {test_image.shape}")
    pixel_count = test_image.size
    
    # Calculate required iterations based on prime number theorem
    # For a number with n digits, probability it's prime is ~1/(ln(10^n)) ≈ 1/(2.3*n)
    # In our case, n is roughly the pixel count (since each pixel becomes part of the number)
    estimated_iterations = int(2.3 * pixel_count)
    # Add safety factor
    max_iterations = estimated_iterations * 4
    
    print(f"Pixel count: {pixel_count}")
    print(f"Estimated iterations needed: {estimated_iterations}")
    print(f"Using max iterations: {max_iterations}")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "debug_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save the test image
    Image.fromarray(test_image).save(os.path.join(output_dir, "test_image.png"))
    
    # Check if the initial image has prime representation
    img_int = image_to_int(test_image)
    is_prime = is_probably_prime(img_int, confidence=10)
    print(f"Initial image integer: {img_int}")
    print(f"Is prime? {is_prime}")
    print(f"Bit length: {img_int.bit_length()}")
    
    # Try dithering the image directly to see if that produces a prime
    dithered_img = dither(test_image)
    dithered_int = image_to_int(dithered_img)
    is_dithered_prime = is_probably_prime(dithered_int, confidence=10)
    print(f"Dithered image integer: {dithered_int}")
    print(f"Is dithered prime? {is_dithered_prime}")
    
    # Try adding noise to see if that helps
    for noise_level in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        print(f"\nTesting noise level: {noise_level}")
        for attempt in range(5):
            noisy_img = add_noise(test_image, noise_level=noise_level)
            dithered_noisy = dither(noisy_img)
            noisy_int = image_to_int(dithered_noisy)
            is_noisy_prime = is_probably_prime(noisy_int, confidence=10)
            print(f"  Attempt {attempt+1}: {'PRIME!' if is_noisy_prime else 'not prime'} ({noisy_int})")
            
            if is_noisy_prime:
                # Verify with high confidence
                high_confidence_check = gmpy2.is_prime(noisy_int, 25)
                print(f"  High confidence check: {high_confidence_check}")
                
                # Save this prime image
                Image.fromarray(dithered_noisy).save(
                    os.path.join(output_dir, f"prime_noise_{noise_level}_attempt_{attempt+1}.png")
                )
                break
    
    # Use the PrimePortraitFinder directly
    print("\n===== USING PRIME PORTRAIT FINDER =====")
    finder = PrimePortraitFinder(image_array=test_image, output_dir=output_dir)
    
    # Modify parameters for debugging
    finder.primality_confidence = 10
    finder.noise_level = 0.2  # Increase noise level for more variation
    
    print(f"Finding prime portrait... (max {max_iterations} iterations)")
    dithered_image, prime_int = finder.find_prime_portrait(max_iterations=max_iterations, verbose=True)
    
    if finder.is_prime:
        print(f"\nSUCCESS! Found a prime portrait after {finder.iterations} iterations")
        print(f"Prime number: {prime_int}")
        print(f"Bit length: {prime_int.bit_length()}")
        print(f"Decimal digits: {len(str(prime_int))}")
        
        # Verify with higher confidence
        high_confidence = gmpy2.is_prime(prime_int, 25)
        print(f"High confidence verification: {high_confidence}")
        
        # Save the results
        saved_files = finder.save_results()
        print(f"Saved results to {output_dir}")
    else:
        print("Failed to find a prime portrait within the maximum iterations")
        
    # Let's try a second attempt with different parameters if we failed
    if not finder.is_prime:
        print("\n===== SECOND ATTEMPT WITH DIFFERENT PARAMETERS =====")
        # Create a different test image - single pixel pattern for simplicity
        tiny_image = np.zeros((2, 2), dtype=np.uint8)
        tiny_image[0,0] = 1
        tiny_image[0,1] = 1
        tiny_image[1,0] = 0
        tiny_image[1,1] = 1  # Binary pattern: 1101 (prime-friendly)
        
        tiny_finder = PrimePortraitFinder(image_array=tiny_image, output_dir=os.path.join(output_dir, "tiny"))
        tiny_finder.primality_confidence = 5  # Lower confidence for speed
        tiny_finder.noise_level = 0.3  # Higher noise for more variation
        
        tiny_max_iterations = int(2.3 * tiny_image.size * 4)  # Similar calculation for tiny image
        print(f"Tiny image shape: {tiny_image.shape}")
        print(f"Tiny max iterations: {tiny_max_iterations}")
        
        dithered_image, prime_int = tiny_finder.find_prime_portrait(max_iterations=tiny_max_iterations, verbose=True)
        
        if tiny_finder.is_prime:
            print(f"\nSUCCESS with tiny image! Found after {tiny_finder.iterations} iterations")
            print(f"Prime number: {prime_int}")
            
            # Save the results
            saved_files = tiny_finder.save_results()

if __name__ == "__main__":
    main() 