#!/usr/bin/env python
"""
Run a series of examples demonstrating MeshIt's gradient control and refinesize parameters.

This script runs test_gradient_refinement.py with different parameter combinations to show:
1. The impact of different gradient values on triangulation quality
2. How refinesize parameters affect mesh refinement near feature points
3. How these parameters work with different point distributions (grid, random, circle)

Usage:
    python run_gradient_examples.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_example(name, command, description):
    """Run a single example with the given parameters"""
    print("-" * 80)
    print(f"Running Example: {name}")
    print(f"Description: {description}")
    print(f"Command: {command}")
    print("-" * 80)
    
    # Create output directory
    output_dir = os.path.join("output", name.replace(" ", "_").lower())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Update command with output directory
    full_command = f"{command} --output-dir {output_dir} --save-all --no-display"
    
    try:
        # Run the command
        process = subprocess.run(full_command, shell=True, check=True, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True)
        
        # Print output
        print(process.stdout)
        
        # Save output to log file
        with open(os.path.join(output_dir, "example_log.txt"), "w") as f:
            f.write(f"Example: {name}\n")
            f.write(f"Description: {description}\n")
            f.write(f"Command: {full_command}\n\n")
            f.write("OUTPUT:\n")
            f.write(process.stdout)
            if process.stderr:
                f.write("\nERRORS:\n")
                f.write(process.stderr)
        
        print(f"Example completed successfully, results saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running example: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")

def main():
    """Run a series of examples demonstrating gradient control and refinesize parameters"""
    # Change to the script's directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    # Check if test_gradient_refinement.py exists
    if not os.path.exists("test_gradient_refinement.py"):
        print("Error: test_gradient_refinement.py not found in the current directory")
        sys.exit(1)
    
    # Create main output directory
    os.makedirs("output", exist_ok=True)
    
    # Example 1: Basic Gradient Control
    run_example(
        "Basic Gradient Control",
        "python test_gradient_refinement.py --num-points 36 --point-type grid --gradients 0.5,1.0,2.0,3.0",
        "Compare different gradient values on a 6x6 grid of points. Lower gradient values produce more uniform triangles."
    )
    
    # Example 2: Feature Points with Grid
    run_example(
        "Feature Points with Grid",
        "python test_gradient_refinement.py --num-points 100 --point-type grid --gradients 1.0,2.0 --feature-points 3 --feature-locations -0.5,-0.5,0,0,0.5,0.5",
        "Demonstrates how feature points refine the mesh locally. Three specific feature points are placed to show the local refinement effect."
    )
    
    # Example 3: Random Points with Varied Feature Sizes
    run_example(
        "Random Points with Varied Feature Sizes",
        "python test_gradient_refinement.py --num-points 50 --point-type random --gradients 1.0,2.0 --feature-points 5",
        "Uses random points with 5 random feature points. Shows how gradient control works with irregular point distributions."
    )
    
    # Example 4: Circle Points with Extreme Gradients
    run_example(
        "Circle Points with Extreme Gradients",
        "python test_gradient_refinement.py --num-points 40 --point-type circle --noise 0.1 --gradients 0.1,1.0,5.0",
        "Demonstrates extreme gradient values on a noisy circle. Gradient 0.1 forces very uniform triangulation, while 5.0 allows significant size variation."
    )
    
    # Example 5: Controlled Base Size
    run_example(
        "Controlled Base Size",
        "python test_gradient_refinement.py --num-points 64 --point-type grid --gradients 1.0 --base-size 0.1 --feature-points 4",
        "Uses a specific base size (0.1) instead of auto-calculation. Shows how base size affects overall triangulation density."
    )
    
    # Example 6: Feature Size Variation
    run_example(
        "Feature Size Variation",
        "python test_gradient_refinement.py --num-points 49 --point-type grid --gradients 1.0 --feature-points 4 --feature-size 0.05",
        "Uses a very small feature size (0.05) to create highly refined areas near feature points."
    )
    
    # Example 7: High-Density Triangulation
    run_example(
        "High-Density Triangulation",
        "python test_gradient_refinement.py --num-points 144 --point-type grid --gradients 0.3,1.0,3.0",
        "High-density point cloud with varying gradients to show how gradient control scales with higher resolution."
    )
    
    print("\nAll examples completed!")
    print(f"Results saved to the 'output' directory")
    print("Each example has its own subdirectory with triangulation visualizations")
    
if __name__ == "__main__":
    main() 