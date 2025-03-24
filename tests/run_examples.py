#!/usr/bin/env python
"""
Example script for running coarse triangulation tests with different parameters.
This script demonstrates how to use test_coarse_triangulation.py with various options.
"""

import os
import subprocess
import sys

# Add the parent directory to the Python path so we can import modules more easily
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

def run_example(name, command, description):
    """Run an example command and print its description."""
    print("\n" + "=" * 80)
    print(f"EXAMPLE: {name}")
    print("-" * 80)
    print(description)
    print("-" * 80)
    print(f"Command: {command}")
    print("-" * 80)
    
    try:
        # Create output directory for this example
        output_dir = f"output/{name.replace(' ', '_').lower()}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Add output directory to command if not already specified
        if "--output-dir" not in command and "-o" not in command:
            command += f" --output-dir {output_dir}"
        
        # Use PYTHONPATH to ensure the parent directory is in the path
        env = os.environ.copy()
        # Use semicolon as separator on Windows, colon on Unix
        path_separator = ';' if sys.platform.startswith('win') else ':'
        env['PYTHONPATH'] = f"{parent_dir}{path_separator}{env.get('PYTHONPATH', '')}"
        
        # Run the command with the updated environment
        subprocess.run(command, shell=True, check=True, env=env)
        print(f"\nResults saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running example: {e}")

def main():
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if test_coarse_triangulation.py exists
    if not os.path.exists("test_coarse_triangulation.py"):
        print("Error: test_coarse_triangulation.py not found in the current directory.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Example 1: Basic test with default parameters
    run_example(
        "Basic Test",
        "python test_coarse_triangulation.py --save-all",
        """
        Basic test with default parameters:
        - 25 points in a grid
        - Gradients: 0.5, 1.0, 2.0, 3.0
        - Results saved to files
        """
    )
    
    # Example 2: Grid points with different gradients
    run_example(
        "Fine Gradients",
        "python test_coarse_triangulation.py --num-points 36 --point-type grid --gradients 0.3 0.7 1.0 1.5 2.0 3.0 --save-all",
        """
        Test with more gradient values to see finer transitions:
        - 36 points in a grid (6x6)
        - Gradients: 0.3, 0.7, 1.0, 1.5, 2.0, 3.0
        - Results saved to files
        """
    )
    
    # Example 3: Random points
    run_example(
        "Random Points",
        "python test_coarse_triangulation.py --num-points 50 --point-type random --save-all",
        """
        Test with random points:
        - 50 randomly distributed points
        - Default gradients: 0.5, 1.0, 2.0, 3.0
        - Results saved to files
        """
    )
    
    # Example 4: Circle with noise
    run_example(
        "Circle with Noise",
        "python test_coarse_triangulation.py --num-points 40 --point-type circle --noise 0.5 --save-all",
        """
        Test with points in a circle with noise:
        - 40 points arranged in a circle
        - Noise level: 0.5 (moderate noise)
        - Default gradients: 0.5, 1.0, 2.0, 3.0
        - Results saved to files
        """
    )
    
    # Example 5: High density grid with extreme gradients
    run_example(
        "High Density",
        "python test_coarse_triangulation.py --num-points 100 --point-type grid --gradients 0.1 1.0 5.0 --save-all",
        """
        Test with high density grid and extreme gradients:
        - 100 points in a grid (10x10)
        - Gradients: 0.1 (very uniform), 1.0 (default), 5.0 (very variable)
        - Results saved to files
        """
    )
    
    # Example 6: Save generated points to file
    points_file = "output/test_points.txt"
    run_example(
        "Save Points",
        f"python test_coarse_triangulation.py --num-points 30 --point-type circle --save-points {points_file} --save-all",
        f"""
        Generate points and save them to a file:
        - 30 points arranged in a circle
        - Points saved to {points_file}
        - Default gradients: 0.5, 1.0, 2.0, 3.0
        - Results saved to files
        """
    )
    
    # Example 7: Load points from file
    run_example(
        "Load Points",
        f"python test_coarse_triangulation.py --input-file {points_file} --gradients 1.0 --save-all",
        f"""
        Load points from a file and test with single gradient:
        - Points loaded from {points_file}
        - Gradient: 1.0 (default)
        - Results saved to files
        """
    )
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)

if __name__ == "__main__":
    main() 