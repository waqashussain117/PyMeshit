#!/usr/bin/env python3
"""
Build script for creating MeshIt executable using PyInstaller
"""

import os
import sys
import subprocess
from pathlib import Path

def build_exe(use_clean=True, debug=False):
    """Build standalone executable for MeshIt GUI"""

    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("Building MeshIt executable...")
    print(f"Working directory: {project_root}")

    # Check if we're in OneDrive or similar cloud storage
    if "OneDrive" in str(project_root) or "Dropbox" in str(project_root) or "Google Drive" in str(project_root):
        print("WARNING: You're building from a cloud-synced directory!")
        print("   This can cause permission errors. Consider building from a local directory.")
        print("   Continuing anyway...\n")

    # Check if we're using the correct conda environment
    import sys
    python_path = sys.executable
    if "conda" not in python_path.lower():
        print("WARNING: You're not using a conda environment Python!")
        print("Tetgen and other dependencies are installed in conda environment.")
        print("Please activate the environment first:")
        print("conda activate PyMeshit")
        print("python build_exe.py --no-clean")
        print("Continuing anyway...\n")

    # PyInstaller should be available in PATH when installed via pip
    pyinstaller_exe = "pyinstaller"

    # Quick verification that PyInstaller is available
    try:
        result = subprocess.run([pyinstaller_exe, "--version"], capture_output=True, text=True, check=True)
        version = result.stdout.strip().split('\n')[-1]  # Get the version line
        print(f"Using PyInstaller: {version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: PyInstaller not found. Please install it with: pip install pyinstaller")
        return False

    # Verify critical dependencies are available
    print("Verifying critical dependencies...")
    try:
        import PySide6
        print(f"PySide6 available: {PySide6.__version__}")
    except ImportError as e:
        print(f"PySide6 not available: {e}")
        return False

    try:
        import pyvista
        print(f"PyVista available: {pyvista.__version__}")
    except ImportError as e:
        print(f"PyVista not available: {e}")
        return False

    try:
        import tetgen
        print("TetGen available")
        # Check if TetGen class is available
        from tetgen.pytetgen import TetGen
        print("TetGen class available")
    except ImportError as e:
        print(f"TetGen not available: {e}")
        print("Make sure you're running this in the correct conda environment:")
        print("  conda activate PyMeshit")
        print("  python build_exe.py --no-clean")
        return False

    try:
        from Pymeshit_workflow_gui import MeshItWorkflowGUI
        print("Main GUI module available")
    except ImportError as e:
        print(f"Main GUI module not available: {e}")
        return False

    try:
        import netCDF4
        print(f"NetCDF4 available: {netCDF4.__version__}")
    except ImportError as e:
        print(f"NetCDF4 not available: {e}")
        print("NetCDF4 is required for data I/O operations.")
        print("Install it with: pip install netCDF4")
        return False

    # PyInstaller command for PySide6 application
    cmd = [
        pyinstaller_exe,
        "--name=PyMeshIt",
        "--windowed",  # No console window
        # "--onefile",  # Removed: causes AV false positives
        "--onedir",   # Directory-based: less likely to trigger AV
        "--noconfirm", # Overwrite existing build
        *(["--clean"] if use_clean else []),  # Clean cache and temporary files (optional)
        "--noupx",    # Don't use UPX compression (can cause issues)
        "--runtime-tmpdir=.",  # Use current directory for temp files
        "--log-level=INFO" if debug else "--log-level=WARN", # More verbose in debug mode
        "--hidden-import=pkg_resources",  # For setuptools
        "--hidden-import=importlib",
        "--hidden-import=importlib.util",
        "--hidden-import=inspect",
        # Main script
        "main.py",
        # Include data files
        "--add-data=resources;resources",
        "--add-data=Pymeshit;Pymeshit",
        # Hidden imports for PySide6
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=PySide6.QtOpenGL",
        "--hidden-import=PySide6.QtOpenGLWidgets",
        "--hidden-import=shiboken6",
        # Hidden imports for essential scientific packages
        "--hidden-import=scipy",
        "--hidden-import=scipy.sparse",
        "--hidden-import=scipy.spatial",
        "--hidden-import=scipy.spatial.distance",
        "--hidden-import=numpy",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.pyplot",
        "--hidden-import=PIL",
        "--hidden-import=pyvista",
        "--hidden-import=pyvista.plotting",
        "--hidden-import=pyvista.utilities",
        "--hidden-import=pyvistaqt",
        "--hidden-import=pyvistaqt.plotting",
        "--hidden-import=pyvistaqt.QtInteractor",
        "--hidden-import=pyvistaqt.background_plotter",
        # Tetgen and triangle with all submodules
        "--hidden-import=tetgen",
        "--hidden-import=tetgen._tetgen",  # Compiled extension
        "--hidden-import=tetgen.pytetgen", # Python module with TetGen class
        "--hidden-import=triangle",
        "--hidden-import=triangle.tri",
        "--hidden-import=triangle.data",
        "--hidden-import=triangle.plot",
        # Additional scientific packages
        "--hidden-import=itertools",
        "--hidden-import=gc",
        "--hidden-import=atexit",
        "--hidden-import=logging",
        "--hidden-import=re",
        "--hidden-import=time",
        "--hidden-import=os",
        "--hidden-import=sys",
        # NetCDF4 for data I/O operations
        "--hidden-import=netCDF4",
        "--hidden-import=netCDF4.Dataset",
        "--hidden-import=netCDF4.Variable",
        # Pymeshit specific imports
        "--hidden-import=Pymeshit",
        "--hidden-import=Pymeshit.intersection_utils",
        "--hidden-import=Pymeshit.tetra_mesh_utils",
        "--hidden-import=Pymeshit.core",
        # Additional meshit dependencies
        "--hidden-import=typing",
        "--hidden-import=collections",
        "--hidden-import=collections.abc",
        # Exclude unnecessary packages to reduce build size
        "--exclude-module=torch",
        "--exclude-module=PyQt5",
        "--exclude-module=torchvision",
        "--exclude-module=torchaudio",
        "--exclude-module=pandas",
        "--exclude-module=pillow",
        "--exclude-module=Image",
        "--exclude-module=opencv-python",
        "--exclude-module=cv2",
        "--exclude-module=opencv",
        "--exclude-module=skimage",
        "--exclude-module=scikit-image",
        "--exclude-module=sklearn",
        "--exclude-module=scikit-learn",
        "--exclude-module=tensorflow",
        "--exclude-module=tf",
        "--exclude-module=keras",
        "--exclude-module=jupyter",
        "--exclude-module=notebook",
        "--exclude-module=ipykernel",
        "--exclude-module=ipython",
        "--exclude-module=flask",
        "--exclude-module=django",
        "--exclude-module=requests",
        "--exclude-module=urllib3",
        "--exclude-module=chardet",
        "--exclude-module=certifi",
        "--exclude-module=pip",
        "--exclude-module=setuptools",
        "--exclude-module=wheel",
        # Exclude CUDA and GPU related packages
        "--exclude-module=cuda",
        "--exclude-module=cudnn",
        "--exclude-module=cupy",
        "--exclude-module=numba",
        "--exclude-module=jax",
        "--exclude-module=jaxlib",
        # Additional exclusions for common bloat
        "--exclude-module=debugpy",
        "--exclude-module=ptvsd",
        "--exclude-module=tqdm",
        "--exclude-module=rich",
        "--exclude-module=click",
        "--exclude-module=pkg_resources",
        # Icon (if available)
        *(["--icon=resources/images/app_logo_small.png"] if os.path.exists("resources/images/app_logo_small.png") else []),
        # Output directory
        "--distpath=release",
        "--workpath=build",
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        print("Starting PyInstaller build process...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build completed successfully!")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Check if exe was created (with --onefile, PyInstaller creates a single executable)
        exe_path = project_root / "release" / "PyMeshIt.exe"
        if exe_path.exists():
            print(f"\nExecutable created: {exe_path}")
            print(f"File size: {exe_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("\nWarning: Executable not found in expected location")
            print(f"Looking for: {exe_path}")
            # List contents of release directory to debug
            release_dir = project_root / "release"
            if release_dir.exists():
                print(f"Contents of release directory: {list(release_dir.iterdir())}")
            else:
                print("Release directory does not exist")

    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

    return True

def clean_build():
    """Clean build artifacts"""
    import shutil
    import time

    dirs_to_remove = ["build", "MeshIt.spec"]
    for item in dirs_to_remove:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                print(f"Removed: {item}")
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not remove {item}: {e}")
                print("This is normal if files are in use or in OneDrive.")
                print("You can manually delete the build directory later.")
                # Try to remove individual files instead
                if os.path.isdir(item):
                    try:
                        # Wait a bit and try again
                        time.sleep(2)
                        shutil.rmtree(item, ignore_errors=True)
                        print(f"Removed: {item} (with ignore_errors)")
                    except:
                        pass

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean_build()
    elif len(sys.argv) > 1 and sys.argv[1] == "--no-clean":
        print("Building without --clean flag to avoid permission issues...")
        success = build_exe(use_clean=False)
        if success:
            print("You can find the executable in the 'release' folder")
        else:
            print("Build failed!")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "--debug":
        print("Building with debug analysis to identify missing imports...")
        success = build_exe(use_clean=False, debug=True)
        if success:
            print("You can find the executable in the 'release' folder")
        else:
            print("Build failed!")
            sys.exit(1)
    else:
        success = build_exe(use_clean=True, debug=False)
        if success:
            print("You can find the executable in the 'release' folder")
        else:
            print("Build failed!")
            sys.exit(1)
