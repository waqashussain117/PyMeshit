# filepath: /c:/Users/Waqas Hussain/OneDrive - Università degli Studi di Milano-Bicocca/Documents/GitHub/PZero/MeshIt-master/meshit/__init__.py
# Import from .core submodule
try:
    from .core import Vector3D, Triangle, MeshItModel
    __all__ = ['Vector3D', 'Triangle', 'MeshItModel']
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import meshit components: {e}")