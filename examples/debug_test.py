#!/usr/bin/env python
"""
Debug test for the triangle_mesh module.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from meshit.triangle_mesh import TriangleWrapper, create_hull_segments

def test_basic_triangulation():
    """Test basic triangulation with detailed debug output."""
    print("\n==== Testing Basic Triangulation ====")
    try:
        print("Generating test points...")
        np.random.seed(42)
        points = np.random.rand(20, 2) * 2 - 1  # Range: -1 to 1
        print(f"Generated {len(points)} points")
        
        print("Creating hull segments...")
        try:
            hull_indices, segments = create_hull_segments(points)
            print(f"Created {len(segments)} hull segments from {len(hull_indices)} hull vertices")
        except Exception as e:
            print(f"Error in create_hull_segments: {e}")
            traceback.print_exc()
            return
        
        print("Creating triangle wrapper...")
        wrapper = TriangleWrapper(gradient=1.0, min_angle=20.0, base_size=0.2)
        
        print("Running triangulation...")
        try:
            result = wrapper.triangulate(points, segments)
            print(f"Triangulation complete with {len(result['triangles'])} triangles")
            print(f"Result vertices: {len(result['vertices'])}, original points: {len(points)}")
        except Exception as e:
            print(f"Error in triangulate: {e}")
            traceback.print_exc()
            return
            
        print("Creating plot...")
        try:
            plt.figure(figsize=(8, 8))
            # Use result['vertices'] instead of points for plotting
            plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], result['triangles'], 'k-')
            
            # Plot original points
            plt.plot(points[:, 0], points[:, 1], 'ro')
            
            # Plot the additional vertices that were added during triangulation
            if len(result['vertices']) > len(points):
                additional_points = result['vertices'][len(points):]
                plt.plot(additional_points[:, 0], additional_points[:, 1], 'bx', 
                        label='Added by Triangle')
                plt.legend()
                
            plt.title(f"Basic Triangulation: {len(result['triangles'])} triangles\n"
                     f"{len(result['vertices'])} vertices ({len(result['vertices'])-len(points)} added)")
            plt.axis('equal')
            plt.grid(True)
            plt.savefig("debug_basic_triangulation.png")
            print("Plot saved to debug_basic_triangulation.png")
        except Exception as e:
            print(f"Error in plotting: {e}")
            traceback.print_exc()
            
        print("Basic triangulation test completed!")
    except Exception as e:
        print(f"Unexpected error in test_basic_triangulation: {e}")
        traceback.print_exc()

def test_feature_refinement():
    """Test feature point refinement with detailed debug output."""
    print("\n==== Testing Feature Refinement ====")
    try:
        print("Generating test points...")
        np.random.seed(42)
        points = np.random.rand(20, 2) * 2 - 1  # Range: -1 to 1
        print(f"Generated {len(points)} points")
        
        print("Creating hull segments...")
        try:
            hull_indices, segments = create_hull_segments(points)
            print(f"Created {len(segments)} hull segments")
        except Exception as e:
            print(f"Error in create_hull_segments: {e}")
            traceback.print_exc()
            return
        
        print("Creating feature points...")
        feature_points = np.array([
            [0.0, 0.0],    # Center
            [0.5, 0.5]     # Upper right
        ])
        feature_sizes = np.array([0.1, 0.08])
        print(f"Created {len(feature_points)} feature points with sizes: {feature_sizes}")
        
        print("Creating wrapper without features...")
        wrapper_no_features = TriangleWrapper(gradient=1.5, base_size=0.3)
        
        print("Running triangulation without features...")
        try:
            result_no_features = wrapper_no_features.triangulate(points, segments)
            print(f"Triangulation without features complete: {len(result_no_features['triangles'])} triangles")
            print(f"Result vertices: {len(result_no_features['vertices'])}")
        except Exception as e:
            print(f"Error in triangulate without features: {e}")
            traceback.print_exc()
            return
        
        print("Creating wrapper with features...")
        wrapper_with_features = TriangleWrapper(gradient=1.5, base_size=0.3)
        
        print("Setting feature points...")
        try:
            wrapper_with_features.set_feature_points(feature_points, feature_sizes)
            print("Feature points set successfully")
        except Exception as e:
            print(f"Error in set_feature_points: {e}")
            traceback.print_exc()
            return
            
        print("Running triangulation with features...")
        try:
            result_with_features = wrapper_with_features.triangulate(points, segments)
            print(f"Triangulation with features complete: {len(result_with_features['triangles'])} triangles")
            print(f"Result vertices: {len(result_with_features['vertices'])}")
        except Exception as e:
            print(f"Error in triangulate with features: {e}")
            traceback.print_exc()
            return
            
        print("Creating plot...")
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Plot without features - use result vertices
            ax1.triplot(result_no_features['vertices'][:, 0], result_no_features['vertices'][:, 1], 
                        result_no_features['triangles'], 'k-')
            
            # Plot original points
            ax1.plot(points[:, 0], points[:, 1], 'ko', markersize=4)
            
            ax1.set_title(f"Without Features: {len(result_no_features['triangles'])} triangles\n"
                         f"{len(result_no_features['vertices'])} vertices")
            ax1.set_aspect('equal')
            ax1.grid(True)
            
            # Plot with features - use result vertices
            ax2.triplot(result_with_features['vertices'][:, 0], result_with_features['vertices'][:, 1], 
                        result_with_features['triangles'], 'k-')
            
            # Plot original points
            ax2.plot(points[:, 0], points[:, 1], 'ko', markersize=4)
            
            # Plot feature points
            for point, size in zip(feature_points, feature_sizes):
                ax2.plot(point[0], point[1], 'r*', markersize=10)
                
            # Plot additional vertices added during refinement
            if len(result_with_features['vertices']) > len(points):
                added_points = result_with_features['vertices'][len(points):]
                ax2.plot(added_points[:, 0], added_points[:, 1], 'bx', markersize=4, 
                        label='Added by Triangle')
                ax2.legend()
                
            ax2.set_title(f"With Features: {len(result_with_features['triangles'])} triangles\n"
                         f"{len(result_with_features['vertices'])} vertices "
                         f"({len(result_with_features['vertices'])-len(points)} added)")
            ax2.set_aspect('equal')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig("debug_feature_refinement.png")
            print("Plot saved to debug_feature_refinement.png")
        except Exception as e:
            print(f"Error in plotting: {e}")
            traceback.print_exc()
            
        print("Feature refinement test completed!")
    except Exception as e:
        print(f"Unexpected error in test_feature_refinement: {e}")
        traceback.print_exc()

def main():
    try:
        print("==== Starting Debug Tests ====")
        test_basic_triangulation()
        test_feature_refinement()
        print("\n==== All Tests Completed ====")
    except Exception as e:
        print(f"Main function error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 