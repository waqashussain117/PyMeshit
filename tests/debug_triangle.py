#!/usr/bin/env python
"""
Simplified debug script for triangle mesh refinement
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_triangle")

try:
    import triangle as tr
    import meshit
    from meshit.triangle_wrapper import TriangleWrapper
    HAS_TRIANGLE = True
    logger.info("Successfully imported triangle and meshit modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    HAS_TRIANGLE = False
    
def simple_gradient_test():
    """Run a simple gradient-based triangulation test"""
    # Create a simple grid of points
    n = 10
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.ravel(), Y.ravel()))
    logger.info(f"Created {len(points)} grid points")
    
    # Create a feature point in the center
    feature_point = np.array([[0.0, 0.0]])
    feature_size = np.array([0.1])
    logger.info(f"Created feature point at {feature_point[0]} with size {feature_size[0]}")
    
    # Define domain boundary using convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    segments = np.array([[i, (i + 1) % len(hull_points)] for i in range(len(hull_points))])
    logger.info(f"Created domain boundary with {len(segments)} segments")
    
    # Create output directory
    os.makedirs("triangle_debug", exist_ok=True)
    
    # Test different gradient values
    for gradient in [1.0, 3.0, 5.0]:
        logger.info(f"Testing gradient = {gradient}")
        
        # Initialize the triangle wrapper
        wrapper = TriangleWrapper(gradient=gradient)
        wrapper.set_feature_points(feature_point, feature_size)
        logger.info("Initialized TriangleWrapper")
        
        # Try direct Delaunay first
        try:
            logger.info("Trying direct Delaunay triangulation")
            tri = Delaunay(points)
            plt.figure(figsize=(8, 8))
            plt.triplot(points[:, 0], points[:, 1], tri.simplices)
            plt.plot(feature_point[:, 0], feature_point[:, 1], 'ro')
            plt.title(f"Delaunay Triangulation - {len(tri.simplices)} triangles")
            plt.savefig(f"triangle_debug/delaunay.png")
            plt.close()
            logger.info(f"Saved Delaunay triangulation plot with {len(tri.simplices)} triangles")
        except Exception as e:
            logger.error(f"Error in Delaunay: {e}")
        
        # Try triangle library
        try:
            logger.info("Trying Triangle library")
            A = dict(vertices=points)
            B = tr.triangulate(A, 'q')
            logger.info(f"Triangle output keys: {list(B.keys())}")
            
            if 'triangles' in B:
                plt.figure(figsize=(8, 8))
                plt.triplot(B['vertices'][:, 0], B['vertices'][:, 1], B['triangles'])
                plt.plot(feature_point[:, 0], feature_point[:, 1], 'ro')
                plt.title(f"Triangle Library - {len(B['triangles'])} triangles")
                plt.savefig(f"triangle_debug/triangle_basic.png")
                plt.close()
                logger.info(f"Saved Triangle library plot with {len(B['triangles'])} triangles")
            else:
                logger.warning("No triangles in Triangle output")
        except Exception as e:
            logger.error(f"Error in Triangle: {e}")
        
        # Try our custom wrapper
        try:
            logger.info("Trying our custom wrapper")
            # Combine domain points with feature point
            all_points = np.vstack([points, feature_point])
            result = wrapper.triangulate(all_points, segments)
            
            if result and 'triangles' in result:
                plt.figure(figsize=(8, 8))
                plt.triplot(result['vertices'][:, 0], result['vertices'][:, 1], result['triangles'])
                plt.plot(feature_point[:, 0], feature_point[:, 1], 'ro')
                plt.title(f"Custom Wrapper (Gradient={gradient}) - {len(result['triangles'])} triangles")
                plt.savefig(f"triangle_debug/custom_gradient_{gradient}.png")
                plt.close()
                logger.info(f"Saved custom wrapper plot with {len(result['triangles'])} triangles")
            else:
                logger.warning("No triangles in custom wrapper result")
        except Exception as e:
            logger.error(f"Error in custom wrapper: {e}")
    
    logger.info("Debug test completed")

if __name__ == "__main__":
    logger.info("Starting debug test")
    simple_gradient_test()
    logger.info("Debug test finished") 