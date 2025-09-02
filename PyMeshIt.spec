# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('resources', 'resources'), ('Pymeshit', 'Pymeshit')],
    hiddenimports=['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'shiboken6', 'scipy', 'scipy.sparse', 'numpy', 'matplotlib', 'matplotlib.pyplot', 'PIL', 'pyvista', 'tetgen', 'triangle'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'pandas', 'pillow', 'Image', 'opencv-python', 'cv2', 'opencv', 'skimage', 'scikit-image', 'sklearn', 'scikit-learn', 'tensorflow', 'tf', 'keras', 'jupyter', 'notebook', 'ipykernel', 'ipython', 'flask', 'django', 'requests', 'urllib3', 'chardet', 'certifi', 'pip', 'setuptools', 'wheel', 'cuda', 'cudnn', 'cupy', 'numba', 'jax', 'jaxlib', 'debugpy', 'ptvsd', 'tqdm', 'rich', 'click', 'pkg_resources'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PyMeshIt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resources\\images\\app_logo_small.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PyMeshIt',
)
