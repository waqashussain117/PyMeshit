# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('resources', 'resources'), ('Pymeshit', 'Pymeshit')],
    hiddenimports=['pkg_resources', 'importlib', 'importlib.util', 'inspect', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'shiboken6', 'scipy', 'scipy.sparse', 'scipy.spatial', 'scipy.spatial.distance', 'numpy', 'matplotlib', 'matplotlib.pyplot', 'PIL', 'pyvista', 'pyvista.plotting', 'pyvista.utilities', 'tetgen', 'tetgen._tetgen', 'tetgen.pytetgen', 'triangle', 'triangle.tri', 'triangle.data', 'triangle.plot', 'itertools', 'gc', 'atexit', 'logging', 're', 'time', 'os', 'sys', 'Pymeshit', 'Pymeshit.intersection_utils', 'Pymeshit.tetra_mesh_utils', 'Pymeshit.core', 'typing', 'collections', 'collections.abc'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'PyQt5', 'torchvision', 'torchaudio', 'pandas', 'pillow', 'Image', 'opencv-python', 'cv2', 'opencv', 'skimage', 'scikit-image', 'sklearn', 'scikit-learn', 'tensorflow', 'tf', 'keras', 'jupyter', 'notebook', 'ipykernel', 'ipython', 'flask', 'django', 'requests', 'urllib3', 'chardet', 'certifi', 'pip', 'setuptools', 'wheel', 'cuda', 'cudnn', 'cupy', 'numba', 'jax', 'jaxlib', 'debugpy', 'ptvsd', 'tqdm', 'rich', 'click', 'pkg_resources'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PyMeshIt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resources\\images\\app_logo_small.png'],
)
