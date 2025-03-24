#!/usr/bin/env python
"""
Special test script demonstrating the MeshIt paper triangulation approach.

This script is designed to recreate the triangulation examples from the paper
with strong refinement near feature points (representing wells, faults, etc.)
to showcase the gradient-based refinement algorithm.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull, Delaunay
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Try to import MeshIt modules
try:
    import meshit
    from meshit.core import Surface, Vector3D, GradientControl
    from meshit import extensions
    try:
        from meshit.triangle_wrapper import TriangleWrapper
        HAS_CUSTOM_WRAPPER = True
    except ImportError:
        HAS_CUSTOM_WRAPPER = False
        print("Custom triangle wrapper not available - run will show standard approach")
        
    print("Successfully imported MeshIt modules.")
    HAS_MESHIT = True
except ImportError as e:
    print(f"Error importing MeshIt modules: {e}")
    print("Falling back to basic triangulation without MeshIt.")
    HAS_MESHIT = False
    HAS_CUSTOM_WRAPPER = False
    

def create_well_feature_points(domain_size=2.0):
    """
    Create features representing wells within the domain
    
    Returns:
        Tuple of (feature_points, feature_sizes)
    """
    # Example: Vertical wells at various locations
    features = []
    
    # Add several well points
    features.append([-0.5, -0.2])    # Well 1
    features.append([0.3, 0.5])      # Well 2 
    features.append([-0.1, -0.6])    # Well 3
    
    # Convert to numpy array
    feature_points = np.array(features)
    
    # Set feature sizes (smaller = more refinement)
    # Wells typically need very small elements around them
    feature_sizes = np.array([0.1, 0.08, 0.12])
    
    return feature_points, feature_sizes

def create_fault_feature_points():
    """
    Create features representing fault lines
    
    Returns:
        Tuple of (feature_points, feature_sizes)
    """
    # Create points along a fault line
    t = np.linspace(0, 1, 5)
    x = -0.7 + 1.4 * t
    y = 0.6 - 1.2 * t
    
    fault_points = np.column_stack([x, y])
    
    # Fault sizes (usually we want moderate refinement along faults)
    fault_sizes = np.ones(len(fault_points)) * 0.15
    
    return fault_points, fault_sizes

def create_domain_points(num_points=100, domain_size=2.0):
    """
    Create points for the domain
    
    Args:
        num_points: Number of points
        domain_size: Size of the square domain
        
    Returns:
        Array of points
    """
    half_size = domain_size / 2
    
    # Create a grid of points with slight random offset to avoid grid artifacts
    n = int(np.sqrt(num_points))
    x = np.linspace(-half_size, half_size, n)
    y = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(x, y)
    
    # Add small random perturbation
    X = X + np.random.uniform(-0.01 * domain_size, 0.01 * domain_size, X.shape)
    Y = Y + np.random.uniform(-0.01 * domain_size, 0.01 * domain_size, Y.shape)
    
    points = np.column_stack((X.ravel(), Y.ravel()))
    
    return points

def compute_convex_hull(points):
    """Compute convex hull of points"""
    hull = ConvexHull(points)
    return points[hull.vertices]

def polygon_area(points):
    """Calculate the area of a polygon"""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def triangulate_with_gradient_control(points, feature_points, feature_sizes, gradient, output_dir="output"):
    """
    Perform triangulation with gradient control
    
    Args:
        points: Array of domain points
        feature_points: Array of feature points
        feature_sizes: Array of feature sizes
        gradient: Gradient control parameter
        
    Returns:
        Dictionary with triangulation results
    """
    if not HAS_CUSTOM_WRAPPER:
        print("Custom triangle wrapper not available")
        return None
    
    # Calculate convex hull to use as boundary
    hull_points = compute_convex_hull(points)
    segments = np.array([[i, (i + 1) % len(hull_points)] for i in range(len(hull_points))])
    
    # Create and configure triangle wrapper
    wrapper = TriangleWrapper(gradient=gradient)
    wrapper.set_feature_points(feature_points, feature_sizes)
    
    # Combine domain points with feature points
    all_points = np.vstack([points, feature_points])
    
    # Perform triangulation
    result = wrapper.triangulate(all_points, segments)
    
    return result

def plot_triangulation_for_paper(domain_points, feature_points, feature_sizes, result, 
                             title, filename, feature_type="well", show_feature_influence=True):
    """
    Create a publication-quality plot of the triangulation similar to the paper
    
    Args:
        domain_points: Original domain points
        feature_points: Feature points
        feature_sizes: Feature sizes
        result: Triangulation result
        title: Plot title
        filename: Output filename
        feature_type: Type of feature (well, fault)
        show_feature_influence: Whether to show feature influence circles
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot triangulation
    if result is not None:
        # Plot triangulation with blue edges
        plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                   result['triangles'], 'b-', alpha=0.6, linewidth=0.7)
        
        # Color triangles by size
        if False:  # Disabled for clarity
            triangles = result['triangles']
            vertices = result['vertices']
            
            # Calculate triangle sizes
            triangle_sizes = []
            for tri in triangles:
                v = vertices[tri]
                # Calculate longest edge
                edges = []
                edges.append(np.linalg.norm(v[1] - v[0]))
                edges.append(np.linalg.norm(v[2] - v[1]))
                edges.append(np.linalg.norm(v[0] - v[2]))
                triangle_sizes.append(max(edges))
                
            # Normalize sizes
            sizes = np.array(triangle_sizes)
            norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
            
            # Plot triangles with color based on size
            for i, tri in enumerate(triangles):
                t = plt.Polygon(vertices[tri], alpha=0.3, 
                               facecolor=plt.cm.viridis(1 - norm_sizes[i]))
                ax.add_patch(t)
    
    # Plot convex hull
    hull_points = compute_convex_hull(domain_points)
    hull_closed = np.vstack((hull_points, hull_points[0]))
    plt.plot(hull_closed[:, 0], hull_closed[:, 1], 'g-', linewidth=2)
    
    # Plot feature points according to their type
    if feature_type == "well":
        # Wells as red circles
        plt.plot(feature_points[:, 0], feature_points[:, 1], 'ro', markersize=8)
        
        # Label wells
        for i, (x, y) in enumerate(feature_points):
            plt.text(x, y + 0.05, f"W{i+1}", fontsize=12, ha='center')
            
    elif feature_type == "fault":
        # Connect fault points with red line
        plt.plot(feature_points[:, 0], feature_points[:, 1], 'r-', linewidth=3)
        
        # Add fault points as red dots
        plt.plot(feature_points[:, 0], feature_points[:, 1], 'ro', markersize=5)
        
        # Label fault
        mid_point = feature_points[len(feature_points)//2]
        plt.text(mid_point[0], mid_point[1] + 0.1, "Fault", color='red', 
                fontsize=12, ha='center')
    
    # Show feature influence
    if show_feature_influence:
        for i, fp in enumerate(feature_points):
            # Inner circle showing feature size
            inner_circle = Circle((fp[0], fp[1]), feature_sizes[i], 
                                  color='r', fill=False, linewidth=1.5, 
                                  alpha=0.7, linestyle='-')
            ax.add_patch(inner_circle)
            
            # Outer circle showing influence radius (approximated from gradient)
            influence_radius = feature_sizes[i] * (1 + 5 * gradient)
            outer_circle = Circle((fp[0], fp[1]), influence_radius, 
                                 color='r', fill=False, linewidth=1, 
                                 alpha=0.3, linestyle='--')
            ax.add_patch(outer_circle)
    
    # Formatting
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f"{title}\n{len(result['triangles'])} triangles", fontsize=14)
    
    # Label axes
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    
    # Add gradient info
    plt.text(0.02, 0.02, f"Gradient = {gradient:.1f}", transform=ax.transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Save figure with high quality
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_paper_example(gradient_values=[0.5, 1.0, 2.0, 5.0], output_dir="meshit_paper_example"):
    """
    Run triangulation examples similar to those in the MeshIt paper
    
    Args:
        gradient_values: List of gradient values to test
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate domain points
    domain_points = create_domain_points(num_points=100)
    
    # 1. Test with well features
    well_points, well_sizes = create_well_feature_points()
    
    print("\n=== Testing Well Features ===")
    for gradient in gradient_values:
        print(f"\nTriangulating with gradient {gradient:.1f}")
        result = triangulate_with_gradient_control(
            domain_points, well_points, well_sizes, gradient, output_dir
        )
        
        if result is not None:
            # Plot the result
            plot_triangulation_for_paper(
                domain_points, well_points, well_sizes, result,
                f"Well Triangulation (Gradient {gradient:.1f})",
                os.path.join(output_dir, f"well_gradient_{gradient:.1f}.png"),
                feature_type="well"
            )
    
    # 2. Test with fault features
    fault_points, fault_sizes = create_fault_feature_points()
    
    print("\n=== Testing Fault Features ===")
    for gradient in gradient_values:
        print(f"\nTriangulating with gradient {gradient:.1f}")
        result = triangulate_with_gradient_control(
            domain_points, fault_points, fault_sizes, gradient, output_dir
        )
        
        if result is not None:
            # Plot the result
            plot_triangulation_for_paper(
                domain_points, fault_points, fault_sizes, result,
                f"Fault Triangulation (Gradient {gradient:.1f})",
                os.path.join(output_dir, f"fault_gradient_{gradient:.1f}.png"),
                feature_type="fault"
            )
    
    # 3. Create a combined example with both wells and faults
    print("\n=== Testing Combined Features ===")
    
    # Combine well and fault features
    combined_points = np.vstack([well_points, fault_points])
    combined_sizes = np.hstack([well_sizes, fault_sizes])
    
    # Use higher gradients for more dramatic effect
    combined_gradients = [2.0, 5.0]
    
    for gradient in combined_gradients:
        print(f"\nTriangulating combined features with gradient {gradient:.1f}")
        result = triangulate_with_gradient_control(
            domain_points, combined_points, combined_sizes, gradient, output_dir
        )
        
        if result is not None:
            # Create a special visualization for the combined case
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot triangulation
            plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], 
                       result['triangles'], 'b-', alpha=0.6, linewidth=0.7)
            
            # Plot convex hull
            hull_points = compute_convex_hull(domain_points)
            hull_closed = np.vstack((hull_points, hull_points[0]))
            plt.plot(hull_closed[:, 0], hull_closed[:, 1], 'g-', linewidth=2)
            
            # Plot wells
            plt.plot(well_points[:, 0], well_points[:, 1], 'ro', markersize=8)
            
            # Plot fault
            plt.plot(fault_points[:, 0], fault_points[:, 1], 'r-', linewidth=3)
            plt.plot(fault_points[:, 0], fault_points[:, 1], 'ro', markersize=5)
            
            # Label features
            for i, (x, y) in enumerate(well_points):
                plt.text(x, y + 0.05, f"W{i+1}", fontsize=12, ha='center')
                
            mid_point = fault_points[len(fault_points)//2]
            plt.text(mid_point[0], mid_point[1] + 0.1, "Fault", color='red', 
                    fontsize=12, ha='center')
            
            # Formatting
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.title(f"Combined Features Triangulation (Gradient {gradient:.1f})\n{len(result['triangles'])} triangles", 
                     fontsize=14)
            
            # Add gradient info
            plt.text(0.02, 0.02, f"Gradient = {gradient:.1f}", transform=ax.transAxes, 
                     fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"combined_gradient_{gradient:.1f}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nPaper examples complete. Results saved to '{output_dir}'")

def main():
    parser = argparse.ArgumentParser(description='Generate MeshIt paper triangulation examples')
    
    parser.add_argument('--gradients', type=str, default='0.5,1.0,2.0,5.0',
                        help='Comma-separated list of gradient values')
    parser.add_argument('--output-dir', type=str, default='meshit_paper_example',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Convert gradient values to list of floats
    gradient_values = [float(val) for val in args.gradients.split(',')]
    
    # Run examples
    run_paper_example(gradient_values, args.output_dir)

if __name__ == '__main__':
    main() 