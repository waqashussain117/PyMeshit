from .core._meshit import (
    MeshItModel,
    Vector3D,
    Surface,
    Polyline,
    create_surface,
    create_polyline,
    Triangle,
    Intersection,
    TriplePoint
)

__version__ = '0.1.1'

# Helper functions for adding geometries to a model
def add_surface_to_model(model, surface):
    """Add a surface to a MeshItModel instance.
    
    Args:
        model: A MeshItModel instance
        surface: A Surface instance to add
    """
    model.surfaces = list(model.surfaces) + [surface]
    
def add_polyline_to_model(model, polyline):
    """Add a polyline to a MeshItModel instance.
    
    Args:
        model: A MeshItModel instance
        polyline: A Polyline instance to add
    """
    model.model_polylines = list(model.model_polylines) + [polyline]

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
    'get_triple_points'
]
