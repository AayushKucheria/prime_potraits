# Prime Portraits

Transform images into visually similar representations that are mathematically prime numbers, based on the work of Zachary Abel (Department of Applied Mathematics, MIT).

## Overview

Prime Portraits creates prime numbers that, when displayed visually, form recognizable images. The algorithm converts images into large integers, then iteratively modifies them until finding one that is a prime number while preserving visual similarity to the original image.

As described in Abel's paper, these primes can be surprisingly intricate - when their digits are arranged in a grid and colored by value, they reveal portraits, logos, or other visual patterns with only mild distortion.

## How It Works

The method relies on two mathematical principles:

1. **Primes are everywhere**: The Prime Number Theorem ensures that among numbers with n digits, approximately 1 in 2.3n is prime.
2. **Primes are easy to spot (with computers)**: Modern primality testing algorithms can efficiently verify if large numbers are prime without needing to factor them.

Our approach:

1. Start with a proper image
2. Use Floyd-Steinberg dithering to approximate the image using a limited set of shades
3. Add subtle noise (±1% per pixel) before dithering to generate different pixelations
4. Convert the resulting grid to a large integer
5. Test for primality using the Miller-Rabin test
6. Repeat until a prime is found

## Notable Examples (from Abel's paper)

- **Mersenne Prime**: A 16,129-digit prime that shows the face of Marin Mersenne when arranged in a grid
- **"Optimus" Prime**: A 12,800-digit prime in base 9
- **Sophie Germain Prime**: A 5,671-digit number that is both visually and mathematically a Sophie Germain prime (where 2p+1 is also prime)
- **Gaussian Prime**: A complex prime number whose real and imaginary parts form visual patterns
- **Twin Primes**: 1,271-digit primes that are both twin primes and "sexy primes"
- **Fermat Spiral Prime**: A 5,000-digit prime arranged along a Fermat spiral

## Requirements

- Python 3.6+
- Dependencies:
  - gmpy2 (for large integer arithmetic and primality testing)
  - matplotlib (for visualization)
  - numpy (for numerical operations)
  - Pillow (for image processing)
  - scikit-image (for dithering)
  - numba (for JIT compilation)
  - psutil (for memory profiling)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prime_portraits.git
cd prime_portraits

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your image in the project directory (default expects 'ea_logo.png')
2. Run the script:

```bash
python main.py
```

3. Results will be saved in the `prime_portrait_output` directory:
   - Original grayscale image
   - Dithered prime image
   - The prime number as a text file
   - Histograms comparing the original and prime images

## Implementation Details

- **Dithering**: Uses Floyd-Steinberg dithering to approximate images with limited color depth
- **Noise Generation**: Adds ±1% random noise to create variations that maintain visual similarity
- **Primality Testing**: Uses Miller-Rabin probabilistic primality test with high confidence settings
- **Optimization**: JIT compilation, parallel processing, and optimized integer conversion for performance

## References

Abel, Z. "Prime Portraits." Bridges Finland Conference Proceedings.

## License

MIT
