# This file is intentionally empty to make the directory a Python package

"""
Core C++ bindings for MeshIt
"""

try:
    from ._meshit import (
        MeshItModel,
        Vector3D,
        Surface,
        Polyline,
        create_surface,
        create_polyline,
        Triangle,
        Intersection,
        TriplePoint,
        GradientControl
    )
except ImportError as e:
    import sys
    print(f"Error importing from _meshit module: {e}", file=sys.stderr)
    print("Available in _meshit:", dir(sys.modules.get('meshit.core._meshit', {})), file=sys.stderr)
    
    # Provide minimal interface to prevent complete package breakage
    from ._meshit import Surface, Vector3D, GradientControl

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
    'GradientControl'
]
