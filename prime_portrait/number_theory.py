"""
Number theory functions for the Prime Portrait package.
"""

import gmpy2
import numpy as np
import hashlib
import multiprocessing
import os

def _process_chunk(chunk, bits_per_pixel=8):
    """Process a single chunk of pixels into a binary integer value."""
    # Convert each pixel to base-256 digit
    result = gmpy2.mpz(0)
    for pixel in chunk:
        result = result * 256 + int(pixel)
    return result

def image_to_int(image, use_mp=True, use_hybrid=True):
    """
    Ultra-fast image to integer conversion using:
    1. Base-256 encoding (treating pixels as bytes in a big number)
    2. Vectorized NumPy operations
    3. Parallel processing for chunks
    4. Optimized binary operations
    5. Hybrid approach to avoid overhead for small images
    
    Args:
        image: A numpy array representing the grayscale image
        use_mp: Whether to use multiprocessing for parallel processing
        use_hybrid: Whether to use a hybrid approach for small images
        
    Returns:
        A gmpy2.mpz integer representing the image
    """
    flat_array = image.flatten()
    total_size = len(flat_array)
    
    # For very small images, use direct method
    if total_size < 1000 or not use_hybrid:
        # Direct method - convert to base-256 number
        result = gmpy2.mpz(0)
        for pixel in flat_array:
            result = result * 256 + int(pixel)
        return result
    
    # For larger images use chunked approach with potential parallelization
    result = gmpy2.mpz(0)
    
    # Using larger chunks for better vectorization
    chunk_size = 512 if use_mp else 256
    num_chunks = (len(flat_array) + chunk_size - 1) // chunk_size
    chunks = [flat_array[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    
    if use_mp and num_chunks > 4:  # Only use multiprocessing if we have enough chunks
        # Get the number of CPUs for parallel processing
        num_processes = min(multiprocessing.cpu_count(), 8)  # Limit to 8 cores max
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process chunks in parallel
            chunk_vals = pool.map(_process_chunk, chunks)
            
            # Combine chunk values into a single large integer
            result = gmpy2.mpz(0)
            for val in chunk_vals:
                result = result * (256 ** len(val)) + val
    else:
        # Serial processing for smaller images or when multiprocessing is disabled
        for chunk in chunks:
            # Process each chunk and combine with the result
            result = result * (256 ** len(chunk)) + _process_chunk(chunk)
    
    return result

# Cache for primality test results
prime_cache = {}

def is_probably_prime(n, confidence=10):
    """
    Fast probabilistic primality test with caching.
    Higher confidence values provide more reliable results but are slower.
    
    Args:
        n: The integer to test for primality
        confidence: The confidence level (more iterations = higher confidence)
        
    Returns:
        Boolean indicating whether the number is probably prime
    """
    # Small numbers can be checked exactly
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Check cache first
    n_hash = str(hash(str(n)))
    if n_hash in prime_cache:
        return prime_cache[n_hash]
    
    # Use Miller-Rabin probabilistic test with specified confidence
    result = gmpy2.is_prime(n, confidence)
    
    # Cache result
    prime_cache[n_hash] = result
    return result

def hash_image(image):
    """
    Get a unique hash for an image to use in caching.
    
    Args:
        image: A numpy array representing the image
        
    Returns:
        A string hash of the image
    """
    h = hashlib.sha256()
    h.update(image.tobytes())
    return h.hexdigest()

def save_int_to_file(number, file_path):
    """
    Saves a large integer to a file efficiently.
    
    Args:
        number: The integer to save
        file_path: The path to save the integer to
    """
    with open(file_path, "w") as f:
        if number == 0:
            f.write("0")
            return
            
        chunks = []
        remaining = number
        
        while remaining > 0:
            chunk = remaining % 10**9
            chunks.append(str(chunk).zfill(9))
            remaining //= 10**9
        
        chunks[-1] = chunks[-1].lstrip('0')
        f.write(''.join(reversed(chunks))) 