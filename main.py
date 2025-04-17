import gmpy2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time
import sys
import psutil
import math
from skimage import img_as_float, img_as_ubyte
from numba import jit
import multiprocessing
from functools import partial, lru_cache
import hashlib
import random

# Numba-accelerated dithering implementation
@jit(nopython=True, fastmath=True)
def _dither_numba_core(img_float, levels):
    """
    Core dithering algorithm accelerated with Numba.
    This runs at near-C speed with JIT compilation.
    """
    height, width = img_float.shape
    result = np.zeros_like(img_float)
    
    # Process each pixel with error diffusion
    for y in range(height):
        for x in range(width):
            # Get the pixel value
            old_pixel = img_float[y, x]
            
            # Find closest palette color by finding the index of the nearest level
            min_dist = 2.0  # Initialize with a value larger than maximum possible distance (which is 1.0)
            best_idx = 0
            for i, level in enumerate(levels):
                dist = abs(level - old_pixel)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            new_pixel = levels[best_idx]
            
            # Update the result
            result[y, x] = new_pixel
            
            # Calculate error
            quant_error = old_pixel - new_pixel
            
            # Distribute error to neighboring pixels
            if x + 1 < width:
                img_float[y, x + 1] += quant_error * 7/16
            if y + 1 < height:
                if x - 1 >= 0:
                    img_float[y + 1, x - 1] += quant_error * 3/16
                img_float[y + 1, x] += quant_error * 5/16
                if x + 1 < width:
                    img_float[y + 1, x + 1] += quant_error * 1/16
    
    return result

def dither(image):
    """
    High-performance dithering using Numba JIT compilation.
    Uses 10 evenly spaced grayscale levels and Floyd-Steinberg error diffusion.
    """
    # Create 10 evenly spaced levels between 0 and 1
    levels = np.linspace(0, 1, 10)
    
    # Convert to float (0-1 range)
    img_float = img_as_float(image.astype(np.float32))
    
    # Use the Numba-optimized core function
    result = _dither_numba_core(img_float.copy(), levels)
    
    # Convert back to uint8 (0-255 range)
    return img_as_ubyte(result)

def _process_chunk(chunk, bits_per_pixel=8):
    """Process a single chunk of pixels into a binary integer value."""
    # Convert values directly to bytes, then to a single integer
    # Using vectorized operations instead of Python loop
    powers = 256 ** np.arange(len(chunk)-1, -1, -1)
    return np.sum(chunk.astype(np.uint64) * powers)

def image_to_int(image, use_mp=True, use_hybrid=True):
    """
    Ultra-fast image to integer conversion using:
    1. Base-256 encoding (treating pixels as bytes in a big number)
    2. Vectorized NumPy operations
    3. Parallel processing for chunks
    4. Optimized binary operations
    5. Hybrid approach to avoid overhead for small images
    """
    flat_array = image.flatten()
    total_size = len(flat_array)
    
    # For very small images, use direct method
    if total_size < 1000 or not use_hybrid:
        # Fast path for small images - directly convert to bytes and then to int
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
            for i, val in enumerate(chunk_vals):
                shift_amount = 8 * chunk_size * (num_chunks - i - 1)
                result |= gmpy2.mpz(val) << shift_amount
    else:
        # Serial processing for smaller images or when multiprocessing is disabled
        for i, chunk in enumerate(chunks):
            # Using vectorized operations for each chunk
            val = _process_chunk(chunk)
            
            # Combine with result using efficient binary operations
            shift_amount = 8 * chunk_size * (num_chunks - i - 1)
            result |= gmpy2.mpz(val) << shift_amount
    
    return result

# Cache for primality test results
prime_cache = {}

def is_probably_prime(n, confidence=10):
    """
    Fast probabilistic primality test with caching.
    Higher confidence values provide more reliable results but are slower.
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
    """Get a unique hash for an image to use in caching."""
    h = hashlib.sha256()
    h.update(image.tobytes())
    return h.hexdigest()

def save_int_to_file(number, file_path):
    """Saves a large integer to a file efficiently."""
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

def visualize_histograms(original_image, dithered_image, save_path=None):
    """Creates histograms comparing original and dithered images."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    axes[0].hist(original_image.flatten(), bins=256, range=(0, 255), color='gray')
    axes[0].set_title('Grayscale Image Histogram')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(dithered_image.flatten(), bins=256, range=(0, 255), color='gray')
    axes[1].set_title('Dithered Image Histogram')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def profile_memory():
    """Returns current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def main():
    # Create output directory
    output_dir = "prime_portrait_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process image
    image = Image.open('ea_logo.png').convert('L')
    
    # Optionally downsample the image to improve performance
    # Uncomment the next line to reduce size by 50%
    # image = image.resize((image.width // 2, image.height // 2), Image.LANCZOS)
    
    image.save(os.path.join(output_dir, 'grayscale_image.png'))
    
    # Convert to numpy array
    np_image = np.array(image)
    
    # Initialize variables for the loop
    iteration = 0
    is_prime = False
    current_image = np_image.copy().astype(float)
    
    # Profile variables
    total_time = 0
    noise_time_total = 0
    dither_time_total = 0
    int_conversion_time_total = 0
    primality_check_time_total = 0
    
    print(f"Initial memory usage: {profile_memory():.2f} MB")
    print(f"Image shape: {np_image.shape}")
    
    # Test optimization strategies
    print("\nTesting integer conversion performance:")
    
    # Test multiprocessing vs serial
    mp_start = time.time()
    _ = image_to_int(np_image, use_mp=True, use_hybrid=True)
    mp_time = time.time() - mp_start
    
    serial_start = time.time()
    _ = image_to_int(np_image, use_mp=False, use_hybrid=True)
    serial_time = time.time() - serial_start
    
    # Choose the faster method for this image size
    use_mp = mp_time < serial_time
    print(f"Multiprocessing: {mp_time:.4f}s, Serial: {serial_time:.4f}s")
    print(f"Using {'multiprocessing' if use_mp else 'serial'} processing")
    
    # Test with and without hybrid approach
    hybrid_start = time.time()
    _ = image_to_int(np_image, use_mp=use_mp, use_hybrid=True)
    hybrid_time = time.time() - hybrid_start
    
    direct_start = time.time()
    _ = image_to_int(np_image, use_mp=use_mp, use_hybrid=False)
    direct_time = time.time() - direct_start
    
    # Choose the faster approach
    use_hybrid = hybrid_time < direct_time
    print(f"Hybrid: {hybrid_time:.4f}s, Direct: {direct_time:.4f}s")
    print(f"Using {'hybrid' if use_hybrid else 'direct'} approach\n")
    
    # Warm up JIT compilation before starting the main loop
    print("Precompiling Numba functions...")
    test_img = np_image.copy()
    _ = dither(test_img)  # First call triggers compilation
    
    # Time the optimized function
    start = time.time()
    dither(test_img)
    dither_time = time.time() - start
    print(f"Optimized dithering time: {dither_time:.4f}s\n")
    
    # Optimize gmpy2 primality testing
    orig_ctx = gmpy2.get_context().copy()
    ctx = gmpy2.get_context()
    ctx.precision = 10000  # Higher precision for large integers
    gmpy2.set_context(ctx)
    
    # Toggle for probabilistic primality testing
    # Increasing confidence makes it more accurate but slower
    primality_confidence = 10  # Default value
    
    # For advanced probabilistic testing: start with fast tests and confirm with more rigorous tests
    early_terminate = True
    
    # Do-while loop until a prime number is found
    while not is_prime:
        iteration += 1
        loop_start = time.time()
        
        if iteration % 100 == 0:
            print(f"\nIteration {iteration}")
            print(f"Memory usage: {profile_memory():.2f} MB")
        
        # Add random noise (±1%)
        noise_start = time.time()
        noise = np.random.uniform(-0.01, 0.01, current_image.shape) * 255
        noisy_image = np.clip(current_image + noise, 0, 255).astype(np.uint8)
        noise_time = time.time() - noise_start
        noise_time_total += noise_time
        
        # Apply dithering
        dither_start = time.time()
        dithered_image = dither(noisy_image)
        dither_time = time.time() - dither_start
        dither_time_total += dither_time
        
        # Convert to integer and check primality
        int_start = time.time()
        image_int = image_to_int(dithered_image, use_mp=use_mp, use_hybrid=use_hybrid)
        int_time = time.time() - int_start
        int_conversion_time_total += int_time
        
        primality_start = time.time()
        
        # First do a quick probabilistic check with low confidence
        if early_terminate and iteration % 10 != 0:  # Only do full test every 10th iteration
            is_prime = is_probably_prime(image_int, confidence=1)
            if is_prime:
                # If it passes the fast test, then verify with higher confidence
                is_prime = gmpy2.is_prime(image_int, primality_confidence)
        else:
            # Do the full test with higher confidence
            is_prime = gmpy2.is_prime(image_int, primality_confidence)
            
        primality_time = time.time() - primality_start
        primality_check_time_total += primality_time
        
        if iteration % 100 == 0:
            print(f"  Noise generation: {noise_time:.4f}s")
            print(f"  Dithering: {dither_time:.4f}s")
            print(f"  Int conversion: {int_time:.4f}s")
            print(f"  Primality check: {primality_time:.4f}s")
            print(f"  Is prime? {is_prime}")
        
        # Use the current dithered image as base for next iteration
        current_image = dithered_image.astype(float)
        
        loop_time = time.time() - loop_start
        total_time += loop_time
    
    # Restore original context
    gmpy2.set_context(orig_ctx)
    
    # Print profiling summary
    print("\n===== PROFILING SUMMARY =====")
    print(f"Total iterations: {iteration}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per iteration: {total_time/iteration:.4f}s")
    print(f"Time breakdown:")
    print(f"  Noise generation: {noise_time_total:.2f}s ({noise_time_total/total_time*100:.1f}%)")
    print(f"  Dithering: {dither_time_total:.2f}s ({dither_time_total/total_time*100:.1f}%)")
    print(f"  Int conversion: {int_conversion_time_total:.2f}s ({int_conversion_time_total/total_time*100:.1f}%)")
    print(f"  Primality check: {primality_check_time_total:.2f}s ({primality_check_time_total/total_time*100:.1f}%)")
    
    # Save the final dithered image
    Image.fromarray(dithered_image).save(os.path.join(output_dir, 'dithered_prime_image.png'))
    
    # Final verification with maximum confidence
    final_check = gmpy2.is_prime(image_int, 50)
    print(f"\nFinal primality check with maximum confidence: {final_check}")
    
    # Save the final prime integer
    save_int_to_file(image_int, os.path.join(output_dir, 'prime_image_int.txt'))
    
    # Create histograms for the original and final dithered image
    visualize_histograms(np_image, dithered_image, os.path.join(output_dir, 'histogram.png'))
    
    print(f"\nSuccess! Found a prime number after {iteration} iterations.")
    print(f"All files saved to '{output_dir}' directory")

if __name__ == "__main__":
    # On Windows, protect the entry point
    if os.name == 'nt':
        multiprocessing.freeze_support()
    main()


    
    
    
    
    
    
    
    
    

    
