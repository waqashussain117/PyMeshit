#!/usr/bin/env python
"""
Script to run the workflow test with different parameters.
This demonstrates the effect of different gradient values on triangulation.
"""

import os
import subprocess
import sys

def run_example(name, command, description):
    """Run an example and print description"""
    print("\n" + "=" * 60)
    print(f"EXAMPLE: {name}")
    print("-" * 60)
    print(description)
    print("-" * 60)
    print(f"Command: {command}")
    print("-" * 60)
    
    # Create output directory
    output_dir = f"output/{name.replace(' ', '_').lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Add output directory to command
    if "--output-dir" not in command:
        command += f" --output-dir {output_dir}"
    
    # Add save-all flag for visualization
    if "--save-all" not in command:
        command += " --save-all"
    
    # Run the command
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Results saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running example: {e}")

def main():
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if the workflow script exists
    if not os.path.exists("test_workflow_simple.py"):
        print("Error: test_workflow_simple.py not found in the current directory.")
        sys.exit(1)
    
    # Create main output directory
    os.makedirs("output", exist_ok=True)
    
    # Example 1: Basic grid with default parameters
    run_example(
        "Basic Grid",
        "python test_workflow_simple.py --num-points 25 --point-type grid --no-display",
        """
        Basic test with grid points:
        - 25 points in a grid pattern
        - Default gradient values: 0.5, 1.0, 2.0, 3.0
        - Shows how gradient affects triangulation
        """
    )
    
    # Example 2: Grid with more gradient values
    run_example(
        "Fine Gradients",
        "python test_workflow_simple.py --num-points 36 --point-type grid --gradients 0.3 0.7 1.0 1.5 2.0 3.0 --no-display",
        """
        Test with more gradient values:
        - 36 points in a grid
        - Gradient values: 0.3, 0.7, 1.0, 1.5, 2.0, 3.0
        - Shows finer transitions between gradients
        """
    )
    
    # Example 3: Random points
    run_example(
        "Random Points",
        "python test_workflow_simple.py --num-points 50 --point-type random --no-display",
        """
        Test with random points:
        - 50 randomly distributed points
        - Default gradient values: 0.5, 1.0, 2.0, 3.0
        - Shows how gradient control works with irregular points
        """
    )
    
    # Example 4: Circle with noise
    run_example(
        "Circle With Noise",
        "python test_workflow_simple.py --num-points 40 --point-type circle --noise 0.5 --no-display",
        """
        Test with circle points with noise:
        - 40 points arranged in a circle with noise
        - Default gradient values: 0.5, 1.0, 2.0, 3.0
        - Shows how gradient control handles noisy boundaries
        """
    )
    
    # Example 5: High density with extreme gradients
    run_example(
        "Extreme Gradients",
        "python test_workflow_simple.py --num-points 100 --point-type grid --gradients 0.1 1.0 5.0 10.0 --no-display",
        """
        Test with extreme gradient values:
        - 100 points in a grid
        - Gradient values: 0.1 (very uniform), 1.0 (standard), 5.0 and 10.0 (very variable)
        - Shows the range of effects possible with gradient control
        """
    )
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nResults are saved in the 'output' directory with subdirectories for each example.")
    print("Each example includes visualizations of:")
    print("1. Input points")
    print("2. Convex hull")
    print("3. Triangulation with different gradient values")

if __name__ == "__main__":
    main() 