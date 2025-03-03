# filepath: /c:/Users/Waqas Hussain/OneDrive - Università degli Studi di Milano-Bicocca/Documents/GitHub/PZero/MeshIt-master/meshit/core/__init__.py
# Import from compiled C++ extension
try:
    from ._meshit import Vector3D, Triangle, MeshItModel
    __all__ = ['Vector3D', 'Triangle', 'MeshItModel']
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import C++ module: {e}")