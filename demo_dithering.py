#!/usr/bin/env python3
"""
Demo script to visualize the dithering functionality from prime_portrait.
Also demonstrates prime portrait generation.
"""

import numpy as np
import matplotlib
# Use non-interactive backend by default
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import datetime
import math
from prime_portrait.image_processing import dither, visualize_histograms, resize_image, resize_for_prime_portrait
from prime_portrait.utils import Timer, profile_memory, add_noise
from prime_portrait.prime_portrait_finder import PrimePortraitFinder

def create_sample_gradient(size=200):
    """Create a sample gradient image to demonstrate dithering."""
    gradient = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        gradient[:, i] = int(i * (255 / size))
    return gradient

def load_image(image_path):
    """Load an image from path and convert to grayscale."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = Image.open(image_path)
    if img.mode != 'L':  # If not already grayscale
        img = img.convert('L')
    return np.array(img)

def create_output_dir():
    """Create a timestamped output directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def estimate_max_iterations(pixel_count):
    """
    Estimate the number of iterations needed to find a prime based on the prime number theorem.
    
    For a number with n digits, the probability it's prime is approximately 1/(2.3n).
    In our case, n is the number of binary digits, which is roughly the pixel count.
    We add a safety factor to increase chances of finding a prime.
    """
    if pixel_count <= 0:
        return 100  # Default fallback
    
    # Apply the prime number theorem: 1/(2.3n) chance of being prime
    estimated_trials = 2.3 * pixel_count
    
    # Add a larger safety factor for larger images
    if pixel_count < 100:
        safety_factor = 4  # Small images
    elif pixel_count < 1000:
        safety_factor = 6  # Medium images
    else:
        safety_factor = 8  # Large images
        
    estimated_trials *= safety_factor
    
    return math.ceil(estimated_trials)

def main():
    parser = argparse.ArgumentParser(description="Demonstrate image dithering and prime portrait generation")
    parser.add_argument("--image", help="Path to input image (optional)")
    parser.add_argument("--size", type=int, default=200, help="Size of sample gradient if no image provided")
    parser.add_argument("--display", action="store_true", help="Try to display images interactively")
    parser.add_argument("--add-noise", action="store_true", help="Add slight noise to the image before dithering")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Noise level (0.01 = 1%)")
    parser.add_argument("--max-iterations", type=int, help="Maximum iterations for prime portrait generation (auto-calculated if not specified)")
    parser.add_argument("--output-dir", help="Specify a custom output directory")
    parser.add_argument("--max-size", type=int, default=512, help="Maximum width/height of the image in pixels")
    parser.add_argument("--target-pixels", type=int, default=None, help="Target number of pixels in the image")
    parser.add_argument("--no-resize", action="store_true", help="Skip the resizing step")
    parser.add_argument("--prime-max-pixels", type=int, default=900, 
                        help="Maximum number of pixels to use for prime portrait (smaller = faster)")
    parser.add_argument("--two-phase-resize", action="store_true", default=True,
                        help="Use two-phase resizing for better quality when drastically reducing size")
    args = parser.parse_args()
    
    # Track memory usage
    initial_memory = profile_memory()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = create_output_dir()
    print(f"Saving results to: {os.path.abspath(output_dir)}")
    
    # Either load provided image or create a sample gradient
    with Timer("Image loading"):
        if args.image:
            try:
                original = load_image(args.image)
                img_name = os.path.splitext(os.path.basename(args.image))[0]
                print(f"Loaded image with shape: {original.shape}")
            except Exception as e:
                print(f"Error loading image: {e}")
                print("Falling back to sample gradient...")
                original = create_sample_gradient(args.size)
                img_name = "gradient"
        else:
            print("Creating sample gradient image...")
            original = create_sample_gradient(args.size)
            img_name = "gradient"
    
    # Resize the image if needed (skip for sample gradient)
    if args.image and not args.no_resize:
        with Timer("Image resizing"):
            orig_pixels = original.shape[0] * original.shape[1]
            
            target_pixels = min(args.target_pixels or 900, 900)  # Default to 900 if not specified
            max_size = args.max_size
            
            # For tiny images, we need to calculate a suitable max_size to achieve ~900 pixels
            # Assuming square aspect ratio as a simple approach: sqrt(900) ≈ 30
            if target_pixels < 1000:
                # Calculate max_size that would result in roughly target_pixels
                # Consider aspect ratio to avoid extreme distortion
                h, w = original.shape
                aspect_ratio = w / h
                
                # Square root of target pixels, adjusted for aspect ratio
                new_height = math.sqrt(target_pixels / aspect_ratio)
                new_width = new_height * aspect_ratio
                
                # Round to nearest integer
                max_size = max(round(new_width), round(new_height))
                
                print(f"Tiny mode enabled: targeting ~{target_pixels} pixels with max dimension {max_size}")
            
            resized = resize_image(
                image_array=original,
                max_size=max_size,
                target_pixels=target_pixels
            )
            
            resized_pixels = resized.shape[0] * resized.shape[1]
            
            reduction_percent = (1 - (resized_pixels / orig_pixels)) * 100
            
            if reduction_percent > 0:
                print(f"Resized image from {original.shape} to {resized.shape}")
                print(f"Pixel reduction: {reduction_percent:.1f}% ({orig_pixels:,} → {resized_pixels:,})")
                
                # Store the resized image for further processing
                original = resized
    
    # Optionally add noise to the image
    if args.add_noise:
        with Timer("Adding noise"):
            print(f"Adding {args.noise_level*100:.1f}% noise to image...")
            original_with_noise = add_noise(original, noise_level=args.noise_level)
        
        # Save both the original and noisy version
        noisy_path = os.path.join(output_dir, f"{img_name}_noisy.png")
        Image.fromarray(original_with_noise).save(noisy_path)
        # Use the noisy image for further processing
        input_image = original_with_noise
    else:
        input_image = original
    
    # Apply dithering
    with Timer("Dithering"):
        print("Applying dithering...")
        dithered = dither(input_image)
    
    # Calculate means for verification
    original_mean = np.mean(original)
    dithered_mean = np.mean(dithered)
    print(f"Original image mean: {original_mean:.2f}")
    print(f"Dithered image mean: {dithered_mean:.2f}")
    print(f"Difference: {abs(original_mean - dithered_mean):.2f}")
    
    # Create histograms
    with Timer("Generating histograms"):
        hist_fig = visualize_histograms(input_image, dithered)
    
    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(dithered, cmap='gray')
    axes[1].set_title("Dithered")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Define output file paths
    comparison_path = os.path.join(output_dir, f"{img_name}_comparison.png")
    histogram_path = os.path.join(output_dir, f"{img_name}_histograms.png")
    original_path = os.path.join(output_dir, f"{img_name}_original.png")
    dithered_path = os.path.join(output_dir, f"{img_name}_dithered.png")
    
    # Save all outputs
    with Timer("Saving outputs"):
        fig.savefig(comparison_path, dpi=300)
        hist_fig.savefig(histogram_path, dpi=300)
        Image.fromarray(original).save(original_path)
        Image.fromarray(dithered).save(dithered_path)
    
    # Generate a prime portrait (now always done)
    prime_output_dir = os.path.join(output_dir, "prime_portrait")
    
    # Use the optimized resize function to make prime finding much faster
    with Timer("Prime-optimized resizing"):
        # First, get the maximum iterations we're willing to do (for UX reasons)
        max_allowed_iterations = 10000 if args.max_iterations is None else args.max_iterations
        
        # Calculate target pixel count that would keep iterations within our limit
        # Since iterations ≈ 2.3 * pixel_count * safety_factor (from estimate_max_iterations)
        # We can work backward: pixel_count ≈ max_iterations / (2.3 * safety_factor)
        # Using a safety factor of 4 as in our enhanced function
        estimated_max_pixels = max_allowed_iterations / (2.3 * 4)
        
        # Use the minimum of user-specified max and our calculated max
        prime_target_pixels = min(args.prime_max_pixels, int(estimated_max_pixels))
        
        # Don't go below 100 pixels (too much quality loss)
        prime_target_pixels = max(100, prime_target_pixels)
        
        print(f"\n===== PREPARING IMAGE FOR PRIME PORTRAIT =====")
        print(f"Original pixel count: {original.shape[0] * original.shape[1]:,}")
        print(f"Target pixel count for prime finding: {prime_target_pixels}")
        
        # Resize specifically for prime portrait generation
        prime_image = resize_for_prime_portrait(
            image_array=input_image,
            target_pixels=prime_target_pixels,
            use_two_phase=args.two_phase_resize
        )
        
        prime_pixels = prime_image.shape[0] * prime_image.shape[1]
        print(f"Resized to {prime_image.shape} = {prime_pixels:,} pixels")
        print(f"Reduction: {(1 - (prime_pixels / (original.shape[0] * original.shape[1]))) * 100:.1f}%")
        
        # Save this prime-optimized image
        Image.fromarray(prime_image).save(os.path.join(output_dir, f"{img_name}_prime_optimized.png"))
    
    # Calculate the pixel count to estimate iterations
    pixel_count = prime_image.size
    
    # Determine max iterations based on prime number theorem if not specified
    if args.max_iterations is None:
        max_iterations = estimate_max_iterations(pixel_count)
    else:
        max_iterations = args.max_iterations
    
    print("\n===== PRIME PORTRAIT GENERATION =====")
    print(f"Image contains {pixel_count:,} pixels")
    print(f"Estimated probability of primality: ~1/{2.3 * pixel_count:.1f}")
    print(f"Starting prime portrait generation (max {max_iterations:,} iterations)...")
    
    with Timer("Prime portrait generation"):
        # Create a PrimePortraitFinder instance
        finder = PrimePortraitFinder(image_array=prime_image, output_dir=prime_output_dir)
        
        # Adjust parameters as needed
        finder.noise_level = args.noise_level
        
        # Find a prime portrait
        dithered_prime, prime_int = finder.find_prime_portrait(
            max_iterations=max_iterations,
            verbose=True
        )
        
        if finder.is_prime:
            # Save the results
            saved_files = finder.save_results()
            
            print("\nPrime portrait generated successfully!")
            print(f"Decimal digits: {len(str(prime_int))}")
            print(f"Binary digits: {prime_int.bit_length()}")
            print(f"Found after {finder.iterations:,} iterations")
            print(f"Results saved to: {os.path.abspath(prime_output_dir)}")
            
            # Create a comparison with the prime portrait
            fig_prime, axes_prime = plt.subplots(1, 3, figsize=(18, 6))
            
            axes_prime[0].imshow(input_image, cmap='gray')
            axes_prime[0].set_title("Original")
            axes_prime[0].axis('off')
            
            axes_prime[1].imshow(dithered, cmap='gray')
            axes_prime[1].set_title("Standard Dithering")
            axes_prime[1].axis('off')
            
            axes_prime[2].imshow(dithered_prime, cmap='gray')
            axes_prime[2].set_title("Prime Portrait")
            axes_prime[2].axis('off')
            
            plt.tight_layout()
            prime_comparison_path = os.path.join(output_dir, f"{img_name}_prime_comparison.png")
            fig_prime.savefig(prime_comparison_path, dpi=300)
        else:
            print("\nNo prime portrait found within the maximum number of iterations.")
            print("Try increasing the maximum iterations or using a smaller image.")
            print("Hint: Use --tiny to resize the image to fewer than 1000 pixels.")
    
    # Save metadata about the run
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        f.write(f"Image: {img_name}\n")
        f.write(f"Original size: {original.shape}\n")
        f.write(f"Prime-optimized size: {prime_image.shape}\n")
        f.write(f"Prime pixel count: {pixel_count:,}\n")
        f.write(f"Original mean: {original_mean:.2f}\n")
        f.write(f"Dithered mean: {dithered_mean:.2f}\n")
        f.write(f"Difference: {abs(original_mean - dithered_mean):.2f}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Memory usage: {profile_memory():.2f} MB\n")
        f.write(f"Estimated probability of primality: ~1/{2.3 * pixel_count:.1f}\n")
        f.write(f"Max iterations: {max_iterations:,}\n")
        f.write(f"Prime found: {finder.is_prime}\n")
        if finder.is_prime:
            f.write(f"Iterations to find prime: {finder.iterations:,}\n")
        if args.image and not args.no_resize:
            f.write(f"Display resizing: enabled\n")
            f.write(f"Max size: {args.max_size}\n")
            f.write(f"Target pixels: {args.target_pixels if args.target_pixels else 'not specified'}\n")
        f.write(f"Prime target pixels: {prime_target_pixels}\n")
        f.write(f"Two-phase resize: {args.two_phase_resize}\n")
    
    print(f"\nSaved output images:")
    print(f"- {os.path.abspath(comparison_path)}")
    print(f"- {os.path.abspath(histogram_path)}")
    print(f"- {os.path.abspath(original_path)}")
    print(f"- {os.path.abspath(dithered_path)}")
    
    # Log final memory usage
    final_memory = profile_memory()
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
    
    # Optionally try to display the images interactively
    if args.display:
        try:
            matplotlib.use('TkAgg')  # Switch to interactive backend
            plt.show()
        except Exception as e:
            print(f"Could not display images interactively: {e}")
            print("Images have been saved to disk instead.")

if __name__ == "__main__":
    main() 