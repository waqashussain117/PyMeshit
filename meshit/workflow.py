"""
MeshIt workflow module.

This module provides functions to run the complete MeshIt workflow:
1. Coarse triangulation
2. Surface-surface intersections
3. Polyline-surface intersections
4. Triple points
5. Align intersections to convex hulls

These steps closely match the MeshIt C++ implementation workflow.
"""

import time
from typing import Optional, Callable, Dict, List, Any, Union
import numpy as np

from .triangle_mesh import TriangleWrapper, create_hull_segments
from .intersection_utils import run_intersection_workflow

def run_complete_workflow(model, gradient: float = 2.0, progress_callback: Optional[Callable] = None):
    """
    Run the complete MeshIt workflow on a model.
    
    Args:
        model: MeshItModel instance
        gradient: Gradient control parameter for triangulation (default: 2.0)
        progress_callback: Optional callback function to report progress
        
    Returns:
        Updated model after processing
    """
    def report_progress(message):
        """Helper function to report progress"""
        if progress_callback:
            progress_callback(message)
        else:
            print(message, end='')
    
    start_time = time.time()
    
    # Step 1: Coarse triangulation
    report_progress(">Start coarse triangulation...\n")
    
    # Process each surface
    n_surfaces = len(model.surfaces)
    for i, surface in enumerate(model.surfaces):
        report_progress(f"   > {(i+1)*100/n_surfaces:.0f}% ({i+1}/{n_surfaces}) {surface.name}\n")
        
        # Ensure convex hull is calculated
        if not hasattr(surface, 'convex_hull') or not surface.convex_hull:
            if hasattr(surface, 'calculate_convex_hull'):
                try:
                    surface.calculate_convex_hull()
                except Exception as e:
                    report_progress(f"   > Error calculating convex hull: {e}\n")
                    continue
        
        # Convert vertices to numpy arrays for triangulation
        vertices_np = np.array([[v.x, v.y, v.z] for v in surface.vertices])
        if len(vertices_np) < 3:
            report_progress(f"   > Surface {surface.name} has fewer than 3 vertices, skipping\n")
            continue
        
        try:
            # Project to 2D for triangulation (assuming z is up)
            # In a real implementation, we'd use the normal to determine the projection plane
            vertices_2d = vertices_np[:, :2]  # Simple 2D projection
            
            # Create boundary segments from convex hull
            if hasattr(surface, 'convex_hull'):
                hull_indices = []
                hull_points_np = []
                
                # Find indices of hull points in vertices array
                for hull_point in surface.convex_hull:
                    for i, vertex in enumerate(surface.vertices):
                        if (abs(hull_point.x - vertex.x) < 1e-10 and
                            abs(hull_point.y - vertex.y) < 1e-10 and
                            abs(hull_point.z - vertex.z) < 1e-10):
                            hull_indices.append(i)
                            hull_points_np.append([vertex.x, vertex.y])
                            break
                
                # Create segments
                segments = np.array([
                    [hull_indices[i], hull_indices[(i+1) % len(hull_indices)]]
                    for i in range(len(hull_indices))
                ])
            else:
                # If no convex hull, compute it from vertices
                _, segments = create_hull_segments(vertices_2d)
            
            # Triangulate with gradient control
            wrapper = TriangleWrapper(gradient=gradient, base_size=0.15)
            result = wrapper.triangulate(vertices_2d, segments)
            
            # Update surface with new triangulation
            surface.triangles = []
            for tri in result['triangles']:
                surface.triangles.append([int(tri[0]), int(tri[1]), int(tri[2])])
                
            report_progress(f"   > Created {len(surface.triangles)} triangles\n")
            
        except Exception as e:
            report_progress(f"   > Error triangulating surface: {e}\n")
    
    report_progress(">...finished\n")
    
    # Step 2-5: Intersection workflow
    run_intersection_workflow(model, progress_callback)
    
    # Report timing
    end_time = time.time()
    elapsed = end_time - start_time
    report_progress(f">Workflow complete. Total time: {elapsed:.2f}s\n")
    
    return model

def run_coarse_triangulation(model, gradient: float = 2.0, progress_callback: Optional[Callable] = None):
    """
    Run only the coarse triangulation step of the workflow.
    
    Args:
        model: MeshItModel instance
        gradient: Gradient control parameter for triangulation (default: 2.0)
        progress_callback: Optional callback function to report progress
        
    Returns:
        Updated model after coarse triangulation
    """
    def report_progress(message):
        """Helper function to report progress"""
        if progress_callback:
            progress_callback(message)
        else:
            print(message, end='')
    
    report_progress(">Start coarse triangulation...\n")
    
    # Process each surface
    n_surfaces = len(model.surfaces)
    for i, surface in enumerate(model.surfaces):
        report_progress(f"   > {(i+1)*100/n_surfaces:.0f}% ({i+1}/{n_surfaces}) {surface.name}\n")
        
        # Ensure convex hull is calculated
        if not hasattr(surface, 'convex_hull') or not surface.convex_hull:
            if hasattr(surface, 'calculate_convex_hull'):
                try:
                    surface.calculate_convex_hull()
                except Exception as e:
                    report_progress(f"   > Error calculating convex hull: {e}\n")
                    continue
        
        # Convert vertices to numpy arrays for triangulation
        vertices_np = np.array([[v.x, v.y, v.z] for v in surface.vertices])
        if len(vertices_np) < 3:
            report_progress(f"   > Surface {surface.name} has fewer than 3 vertices, skipping\n")
            continue
        
        try:
            # Project to 2D for triangulation (assuming z is up)
            vertices_2d = vertices_np[:, :2]  # Simple 2D projection
            
            # Create boundary segments from convex hull
            if hasattr(surface, 'convex_hull'):
                hull_indices = []
                
                # Find indices of hull points in vertices array
                for hull_point in surface.convex_hull:
                    for i, vertex in enumerate(surface.vertices):
                        if (abs(hull_point.x - vertex.x) < 1e-10 and
                            abs(hull_point.y - vertex.y) < 1e-10 and
                            abs(hull_point.z - vertex.z) < 1e-10):
                            hull_indices.append(i)
                            break
                
                # Create segments
                segments = np.array([
                    [hull_indices[i], hull_indices[(i+1) % len(hull_indices)]]
                    for i in range(len(hull_indices))
                ])
            else:
                # If no convex hull, compute it from vertices
                _, segments = create_hull_segments(vertices_2d)
            
            # Triangulate with gradient control
            wrapper = TriangleWrapper(gradient=gradient, base_size=0.15)
            result = wrapper.triangulate(vertices_2d, segments)
            
            # Update surface with new triangulation
            surface.triangles = []
            for tri in result['triangles']:
                surface.triangles.append([int(tri[0]), int(tri[1]), int(tri[2])])
                
            report_progress(f"   > Created {len(surface.triangles)} triangles\n")
            
        except Exception as e:
            report_progress(f"   > Error triangulating surface: {e}\n")
    
    report_progress(">...finished\n")
    
    return model 