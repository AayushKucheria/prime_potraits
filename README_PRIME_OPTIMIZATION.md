# Prime Portrait Optimization

## Problem Summary
The original implementation of the prime portrait generator was not finding prime numbers before reaching maximum iterations, especially with larger images.

## Analysis of the Issue
After testing, we discovered several key issues:

1. **Image Size**: The number of pixels in an image directly affects how many iterations are needed to find a prime. According to the prime number theorem, a number with n digits has approximately a 1/(2.3*n) chance of being prime. For an image with 40,000 pixels, the chance of finding a prime is roughly 1 in 92,000.

2. **Iteration Estimation**: The original safety factor of 2 was insufficient for larger images, leading to premature termination before a prime could be found.

## Solutions Implemented

### 1. Enhanced Image Resizing for Prime Finding
We implemented a new specialized function `resize_for_prime_portrait()` that:

- Precisely targets a specific pixel count (default 900)
- Uses a two-phase compression approach for better quality when drastically reducing size
- Preserves aspect ratio while ensuring a small enough image for efficient prime finding
- Adds flexibility with an optional maximum dimension constraint

```python
# Example usage:
prime_image = resize_for_prime_portrait(
    image_array=original_image,
    target_pixels=500,  # Target 500 pixels total
    use_two_phase=True  # Better quality for drastic reductions
)
```

### 2. Improved Iteration Calculation
We enhanced the `estimate_max_iterations()` function to use a dynamic safety factor based on image size:

```python
# For small images (< 100 pixels): safety factor of 4
# For medium images (< 1000 pixels): safety factor of 6
# For large images (>= 1000 pixels): safety factor of 8
```

### 3. Pre-calculating Maximum Pixel Count
The updated demo script now automatically calculates the ideal target pixel count based on:
- User-specified maximum iterations
- The prime probability formula (iterations ≈ 2.3 * pixel_count * safety_factor)
- A minimum threshold to preserve visual quality

## Results
- With a 100-pixel image, we found a prime after only 276 iterations
- The prime had 238 decimal digits and 789 binary digits
- Processing time was dramatically reduced from several minutes to just seconds

## Usage Recommendations

1. For fastest prime finding: Use `--prime-max-pixels 100` 
2. For balance of quality and speed: Use `--prime-max-pixels 400`
3. For highest quality (but slowest): Use `--prime-max-pixels 900`

## Technical Details
The two-phase compression approach:
1. First reduces to an intermediate size (~50% of original) using high-quality Lanczos resampling
2. Then further reduces to the target pixel count

This preserves more edge details and avoids the blurring artifacts that can occur with a single aggressive resize operation.

## Example Command
```bash
python demo_dithering.py --image your_image.jpg --prime-max-pixels 400 --two-phase-resize
```

This will create a prime portrait with approximately 400 pixels, balancing quality and prime-finding speed. 