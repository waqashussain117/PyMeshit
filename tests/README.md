# Triangulation Gradient Control Tests

This directory contains test scripts for experimenting with gradient control and refinesize parameters during the coarse triangulation process in mesh generation.

## Overview

The main test script `test_coarse_triangulation_simplified.py` demonstrates how different gradient values affect triangulation quality with various point patterns. The script can generate different types of input points (grid, random, circle) and allows you to test multiple gradient values at once, visualizing the results and providing performance metrics.

## Scripts

- **test_coarse_triangulation_simplified.py**: The main test script for running individual triangulation tests
- **run_examples_simplified.py**: A script that runs a series of example tests with different parameters

## Installation

These tests require the following Python packages:
- numpy
- matplotlib
- scipy
- pyvista (optional, for better 3D visualization)

Install them using:
```
pip install numpy matplotlib scipy pyvista
```

## Usage

### Running Individual Tests

Use `test_coarse_triangulation_simplified.py` to run specific tests with custom parameters:

```
python test_coarse_triangulation_simplified.py [OPTIONS]
```

#### Options:

- `--num-points`, `-n`: Number of points to generate (default: 25)
- `--point-type`, `-t`: Type of points to generate (grid, random, circle) (default: grid)
- `--noise`: Noise level for point generation (default: 0.0)
- `--gradients`, `-g`: Gradient values to test (default: 0.5 1.0 2.0 3.0)
- `--input-file`, `-i`: Load points from a file instead of generating them
- `--save-points`, `-s`: Save generated points to a file
- `--output-dir`, `-o`: Output directory for saved files (default: output)
- `--save-all`: Save all figures and data
- `--no-display`: Do not display figures interactively (save only)

### Example Commands

#### Basic usage:
```
python test_coarse_triangulation_simplified.py --save-all
```

#### Generate a denser grid with specific gradients:
```
python test_coarse_triangulation_simplified.py --num-points 100 --gradients 0.1 1.0 5.0 --save-all
```

#### Test with a circle pattern with some noise:
```
python test_coarse_triangulation_simplified.py --point-type circle --num-points 40 --noise 0.5 --save-all
```

### Running All Examples

Use `run_examples_simplified.py` to run a series of predefined examples:

```
python run_examples_simplified.py
```

This will execute all the example tests and save the results in the `output` directory.

## Understanding the Results

For each test, the script will produce:

1. Visualizations of:
   - Input points
   - Triangulations with different gradient values
   
2. A summary CSV file with:
   - Triangle counts for each gradient
   - Processing times
   - Percentage change in triangle count relative to the baseline gradient (1.0)

3. A detailed log file with:
   - Test parameters
   - Feature points
   - Results for each gradient

## Understanding Gradient Control

The gradient control parameter affects how triangle size varies across the mesh:

- **Low gradient values (< 1.0)**: Result in more uniform triangulation with smaller triangles throughout
- **Gradient = 1.0**: Standard triangulation with moderate size variation
- **High gradient values (> 1.0)**: Allow for more size variation, with smaller triangles near feature points/boundaries and larger triangles elsewhere

These tests help demonstrate how gradient values affect triangle count and distribution, allowing you to determine optimal values for your specific use case.

## Adding Your Own Points

To test with your own set of points:

1. Save your points in a text file with one point per line (x y z)
2. Run the test script with the `--input-file` parameter:
   ```
   python test_coarse_triangulation_simplified.py --input-file your_points.txt --save-all
   ```

## Troubleshooting

- If visualization doesn't work properly, make sure PyVista is installed correctly.
- If PyVista is not available, the script will fall back to Matplotlib for basic visualizations.
- For better visualizations, run with the `--save-all` flag and examine the saved PNG files. 