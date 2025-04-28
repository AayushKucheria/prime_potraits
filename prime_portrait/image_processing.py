"""
Image processing functions for the Prime Portrait package.
"""

import numpy as np
from numba import jit
from skimage import img_as_float, img_as_ubyte
from PIL import Image
import matplotlib.pyplot as plt
import math

# Numba-accelerated dithering implementation
@jit(nopython=True, fastmath=True)
def _dither_numba_core(image, levels):
    """
    Core dithering algorithm accelerated with Numba.
    Works directly with uint8 values (0-255) for better intensity preservation.

    Parameters:
        image: A numpy array of uint8 values representing the grayscale image
        levels: An array of quantization levels (uint8 values)
        
    Returns:
        A numpy array with the same shape as image, containing the dithered image
    """
    height, width = image.shape
    result = np.zeros_like(image)
    # Create a buffer for error diffusion (needs to be float for fractional errors)
    buffer = image.astype(np.float32)
    
    # Process each pixel with error diffusion
    for y in range(height):
        for x in range(width):
            # Get the pixel value
            old_pixel = int(buffer[y, x])
            
            # Find closest palette color
            min_dist = 256  # Larger than max possible distance
            best_level = 0
            for level in levels:
                dist = abs(level - old_pixel)
                if dist < min_dist:
                    min_dist = dist
                    best_level = level
            
            # Update the result
            result[y, x] = best_level
            
            # Calculate error
            quant_error = buffer[y, x] - best_level
            
            # Distribute error to neighboring pixels (Floyd-Steinberg pattern)
            if x + 1 < width:
                buffer[y, x + 1] += quant_error * 7/16
            if y + 1 < height:
                if x - 1 >= 0:
                    buffer[y + 1, x - 1] += quant_error * 3/16
                buffer[y + 1, x] += quant_error * 5/16
                if x + 1 < width:
                    buffer[y + 1, x + 1] += quant_error * 1/16
    
    return result

def dither(image):
    """
    High-performance dithering using Numba JIT compilation.
    Uses 10 evenly spaced grayscale levels and Floyd-Steinberg error diffusion.
    
    Args:
        image: A numpy array representing the grayscale image
        
    Returns:
        A numpy array representing the dithered image
    """
    # Ensure the image is uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Create 10 evenly spaced levels in the 0-255 range
    levels = np.linspace(0, 255, 10, dtype=np.uint8)
    
    # Dither directly with uint8 values
    return _dither_numba_core(image, levels)

def resize_image(image_array=None, image_path=None, max_size=None, target_pixels=None, 
                 resample=Image.LANCZOS):
    """
    Resize an image to meet size constraints while preserving aspect ratio.
    At least one of image_array or image_path must be provided.
    
    Args:
        image_array: A numpy array representing the grayscale image
        image_path: Path to an image file
        max_size: Maximum width or height in pixels
        target_pixels: Target number of total pixels
        resample: PIL resampling filter, defaults to high-quality LANCZOS
        
    Returns:
        A numpy array representing the resized grayscale image
    """
    # Validate inputs
    if image_array is None and image_path is None:
        raise ValueError("Either image_array or image_path must be provided")
    
    if image_array is not None and image_path is not None:
        raise ValueError("Only one of image_array or image_path should be provided")
    
    if max_size is not None and max_size <= 0:
        raise ValueError("max_size must be positive")
    
    if target_pixels is not None and target_pixels <= 0:
        raise ValueError("target_pixels must be positive")
    
    # If neither constraint is provided, just return the original
    if max_size is None and target_pixels is None:
        return image_array if image_array is not None else np.array(Image.open(image_path).convert('L'))
    
    # Load the image
    if image_path is not None:
        img = Image.open(image_path)
        # Convert to grayscale if it's not already
        if img.mode != 'L':
            img = img.convert('L')
    else:
        img = Image.fromarray(image_array)
        if img.mode != 'L':
            img = img.convert('L')
    
    # Get original dimensions
    orig_width, orig_height = img.size
    
    # Determine scaling factor based on constraints
    scale_factor = 1.0  # Default: no scaling
    
    if max_size is not None:
        # Scale to fit within max_size while preserving aspect ratio
        if orig_width > max_size or orig_height > max_size:
            scale_factor = min(max_size / orig_width, max_size / orig_height)
    
    if target_pixels is not None:
        # Calculate scaling based on target pixel count
        curr_pixels = orig_width * orig_height
        if curr_pixels > target_pixels:
            pixels_scale_factor = math.sqrt(target_pixels / curr_pixels)
            # Use the more aggressive scaling between max_size and target_pixels
            if max_size is None or pixels_scale_factor < scale_factor:
                scale_factor = pixels_scale_factor
    
    # If no scaling needed (image already meets constraints)
    if scale_factor >= 1.0:
        return np.array(img)
    
    # Calculate new dimensions
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    
    # Ensure at least 1x1
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    # Resize the image
    resized_img = img.resize((new_width, new_height), resample=resample)
    
    # Return as numpy array
    return np.array(resized_img)

def resize_for_prime_portrait(image_array=None, image_path=None, target_pixels=900, 
                             max_dimension=None, use_two_phase=True,
                             intermediate_scale=0.5, resample=Image.LANCZOS):
    """
    Advanced image resizing optimized for prime portrait generation.
    Precisely targets a specific pixel count to make prime finding much faster.
    
    Features:
    - Two-phase compression for better quality when drastically reducing size
    - Precise target pixel count calculation
    - Aspect ratio preservation
    - Optional maximum dimension constraint
    
    Args:
        image_array: A numpy array representing the grayscale image
        image_path: Path to an image file
        target_pixels: Target number of total pixels (default 900, good for prime finding)
        max_dimension: Optional maximum width or height constraint
        use_two_phase: Whether to use two-phase resizing for better quality
        intermediate_scale: Scale factor for the first phase (as fraction of original)
        resample: PIL resampling filter, defaults to high-quality LANCZOS
        
    Returns:
        A numpy array representing the resized grayscale image with ~target_pixels total
    """
    # Validate inputs
    if image_array is None and image_path is None:
        raise ValueError("Either image_array or image_path must be provided")
    
    if image_array is not None and image_path is not None:
        raise ValueError("Only one of image_array or image_path should be provided")
    
    if target_pixels <= 0:
        raise ValueError("target_pixels must be positive")
        
    # Load the image
    if image_path is not None:
        img = Image.open(image_path)
        # Convert to grayscale if it's not already
        if img.mode != 'L':
            img = img.convert('L')
    else:
        img = Image.fromarray(image_array)
        if img.mode != 'L':
            img = img.convert('L')
    
    # Get original dimensions
    orig_width, orig_height = img.size
    orig_pixels = orig_width * orig_height
    aspect_ratio = orig_width / orig_height
    
    # If the image is already smaller than target, no need to resize
    if orig_pixels <= target_pixels and (max_dimension is None or 
                                        (orig_width <= max_dimension and 
                                         orig_height <= max_dimension)):
        return np.array(img)
    
    # Apply max_dimension constraint if specified
    max_dim_scale_factor = 1.0
    if max_dimension is not None:
        if orig_width > max_dimension or orig_height > max_dimension:
            max_dim_scale_factor = min(max_dimension / orig_width, 
                                      max_dimension / orig_height)
    
    # Calculate dimensions to achieve target_pixels while preserving aspect ratio
    # If we have width = height * aspect_ratio, and width * height = target_pixels,
    # then height * aspect_ratio * height = target_pixels
    # So height = sqrt(target_pixels / aspect_ratio)
    
    # Calculate ideal dimensions for target pixel count
    ideal_height = math.sqrt(target_pixels / aspect_ratio)
    ideal_width = ideal_height * aspect_ratio
    
    # Apply max_dimension constraint if needed
    if max_dimension is not None:
        # Check if ideal dimensions exceed max_dimension
        if ideal_width > max_dimension or ideal_height > max_dimension:
            # Scale down to fit max_dimension
            constrained_scale = min(max_dimension / ideal_width, 
                                  max_dimension / ideal_height)
            ideal_width *= constrained_scale
            ideal_height *= constrained_scale
    
    # Round to integers
    final_width = max(1, round(ideal_width))
    final_height = max(1, round(ideal_height))
    
    # Two-phase resize for better quality when shrinking dramatically
    if use_two_phase and orig_pixels > target_pixels * 4:  # When shrinking by >75%
        # First phase: intermediate size using default scaling
        if max_dimension is not None:
            # Intermediate phase should still respect max_dimension
            interm_scale = min(intermediate_scale, max_dim_scale_factor)
        else:
            interm_scale = intermediate_scale
            
        interm_width = max(final_width, round(orig_width * interm_scale))
        interm_height = max(final_height, round(orig_height * interm_scale))
        
        # First resize to intermediate size
        intermediate_img = img.resize((interm_width, interm_height), 
                                     resample=resample)
        
        # Second resize to final target size
        final_img = intermediate_img.resize((final_width, final_height), 
                                          resample=resample)
    else:
        # Single-phase resize directly to target size
        final_img = img.resize((final_width, final_height), resample=resample)
    
    # Return as numpy array
    return np.array(final_img)

def visualize_histograms(original_image, dithered_image, save_path=None):
    """
    Creates histograms comparing original and dithered images.
    
    Args:
        original_image: A numpy array representing the original grayscale image
        dithered_image: A numpy array representing the dithered image
        save_path: Optional path to save the histogram image
        
    Returns:
        A matplotlib figure object containing the histograms
    """
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