"""
MeshIt - Python bindings for the MeshIt C++ library
"""
import sys
try:
    from .core._meshit import (
        MeshItModel,
        Vector3D,
        Surface,
        Polyline,
        create_surface,
        create_polyline,
        Triangle,
        Intersection,
        TriplePoint,
        GradientControl,
    )
except ImportError as e:
    import sys
    print(f"Warning: Could not import directly from core._meshit: {e}", file=sys.stderr)
    
    # Try to import from core module instead
    try:
        from .core import (
            MeshItModel,
            Vector3D,
            Surface,
            Polyline,
            create_surface,
            create_polyline,
            Triangle,
            Intersection,
            TriplePoint,
            GradientControl,
        )
    except ImportError as e2:
        print(f"Error: Failed to import MeshIt core components: {e2}", file=sys.stderr)
        print("MeshIt will have limited functionality", file=sys.stderr)
        
        # Define some minimal classes to prevent complete package breakage
        class MeshItModel:
            def __init__(self):
                self.surfaces = []
                self.model_polylines = []
                print("Warning: Using dummy MeshItModel class - MeshIt core could not be loaded")
            
            def __str__(self):
                return "MeshItModel(dummy)"

# Try to import extensions, but don't fail if it can't find MeshItModel
try:
    from . import extensions
except ImportError as e:
    print(f"Warning: Could not import extensions module: {e}", file=sys.stderr)

from . import core
from . import extensions

# Try to import our direct Triangle wrapper with C++ callback
try:
    from . import triangle_callback
    from . import triangle_direct
    HAS_DIRECT_TRIANGLE = True
except ImportError:
    HAS_DIRECT_TRIANGLE = False
    
# Try to import our Python-based triangle wrapper
try:
    from . import triangle_wrapper
    HAS_TRIANGLE_WRAPPER = True
except ImportError:
    HAS_TRIANGLE_WRAPPER = False

# Define version
__version__ = '0.1.1'

# Helper functions for adding geometries to a model
def add_surface_to_model(model, surface):
    """Add a surface to a MeshItModel instance.
    
    Args:
        model: A MeshItModel instance
        surface: A Surface instance to add
    """
    try:
        model.surfaces = list(model.surfaces) + [surface]
    except Exception as e:
        print(f"Warning: Failed to add surface to model: {e}", file=sys.stderr)
    
def add_polyline_to_model(model, polyline):
    """Add a polyline to a MeshItModel instance.
    
    Args:
        model: A MeshItModel instance
        polyline: A Polyline instance to add
    """
    try:
        model.model_polylines = list(model.model_polylines) + [polyline]
    except Exception as e:
        print(f"Warning: Failed to add polyline to model: {e}", file=sys.stderr)

# Helper functions for accessing model results
def get_intersections(model):
    """Get the intersections from a MeshItModel instance.
    
    Args:
        model: A MeshItModel instance
    
    Returns:
        A list of Intersection objects or an empty list if not available
    """
    try:
        return model.intersections
    except AttributeError:
        return []

def get_triple_points(model):
    """Get the triple points from a MeshItModel instance.
    
    Args:
        model: A MeshItModel instance
    
    Returns:
        A list of TriplePoint objects or an empty list if not available
    """
    try:
        return model.triple_points
    except AttributeError:
        return []

def compute_convex_hull(points):
    """Compute the convex hull of a set of 3D points.
    
    Args:
        points: A list of 3D points, where each point is a list of 3 coordinates [x, y, z]
    
    Returns:
        A list of 3D points representing the convex hull
    """
    try:
        # Create a surface from the points
        surface = create_surface(points, [], "TempSurface", "Scattered")
        
        # Calculate the convex hull
        surface.calculate_convex_hull()
        
        # Convert the convex hull points to a list of lists
        hull_points = []
        for point in surface.convex_hull:
            hull_points.append([point.x, point.y, point.z])
        
        return hull_points
    except Exception as e:
        print(f"Warning: Failed to compute convex hull: {e}", file=sys.stderr)
        
        # Fall back to scipy for convex hull if available
        try:
            import numpy as np
            from scipy.spatial import ConvexHull
            
            points_array = np.array(points)
            if points_array.shape[1] == 3:  # 3D points
                hull = ConvexHull(points_array)
                return points_array[hull.vertices].tolist()
            return []
        except ImportError:
            return []

__all__ = [
    'MeshItModel',
    'Vector3D',
    'Surface',
    'Polyline',
    'create_surface',
    'create_polyline',
    'Triangle',
    'Intersection',
    'TriplePoint',
    'add_surface_to_model',
    'add_polyline_to_model',
    'get_intersections',
    'get_triple_points',
    'compute_convex_hull'
]
