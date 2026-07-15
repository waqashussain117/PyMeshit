# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all

datas = [('resources', 'resources'), ('Pymeshit', 'Pymeshit')]
binaries = []
hiddenimports = ['pkg_resources', 'importlib', 'importlib.util', 'inspect', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets', 'shiboken6', 'scipy', 'scipy.sparse', 'scipy.spatial', 'scipy.spatial.distance', 'numpy', 'matplotlib', 'matplotlib.pyplot', 'PIL', 'pyvista', 'pyvista.plotting', 'pyvista.utilities', 'pyvistaqt', 'pyvistaqt.plotting', 'pyvistaqt.QtInteractor', 'pyvistaqt.background_plotter', 'pooch', 'platformdirs', 'requests', 'urllib3', 'charset_normalizer', 'idna', 'vtkmodules', 'vtkmodules.qt', 'vtkmodules.qt.QVTKRenderWindowInteractor', 'tetgen', 'tetgen._tetgen', 'tetgen.pytetgen', 'triangle', 'triangle.tri', 'triangle.data', 'triangle.plot', 'itertools', 'gc', 'atexit', 'logging', 're', 'time', 'os', 'sys', 'netCDF4', 'netCDF4.utils', 'cftime', 'certifi', 'h5py', 'hdf5plugin', 'Pymeshit', 'Pymeshit.intersection_utils', 'Pymeshit.tetra_mesh_utils', 'Pymeshit.core', 'typing', 'collections', 'collections.abc']
binaries += collect_dynamic_libs('netCDF4')
tmp_ret = collect_all('vtkmodules')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('vtk')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('netCDF4')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cftime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['trimesh', 'gmsh', 'pygmsh', 'meshio', 'pymeshfix', 'shapely', 'rtree', 'nbformat', 'parso', 'tornado', 'zmq', 'pyvista.examples', 'torch', 'PyQt5', 'torchvision', 'torchaudio', 'pandas', 'opencv-python', 'cv2', 'opencv', 'skimage', 'scikit-image', 'sklearn', 'scikit-learn', 'tensorflow', 'tf', 'keras', 'jupyter', 'notebook', 'ipykernel', 'ipython', 'flask', 'django', 'pip', 'setuptools', 'wheel', 'cuda', 'cudnn', 'cupy', 'numba', 'jax', 'jaxlib', 'debugpy', 'ptvsd', 'tqdm', 'rich', 'click', 'pkg_resources', 'IPython', 'jedi', 'tkinter', 'lxml', 'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtPdf', 'PySide6.QtVirtualKeyboard', 'PySide6.QtNetwork', 'PySide6.QtWebEngine', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineWidgets'],
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
