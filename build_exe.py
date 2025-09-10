#!/usr/bin/env python3
"""
Build script for creating MeshIt executable using PyInstaller
"""

import os
import sys
import subprocess
from pathlib import Path

def build_exe():
    """Build standalone executable for MeshIt GUI"""

    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("Building MeshIt executable...")
    print(f"Working directory: {project_root}")

    # Try to find the correct Python environment with PyInstaller
    pyinstaller_paths = [
        r"C:\Users\Waqas Hussain\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\pyinstaller.exe",
        "pyinstaller.exe",
        sys.executable.replace("python.exe", "Scripts\\pyinstaller.exe")
    ]

    pyinstaller_exe = None
    for path in pyinstaller_paths:
        if os.path.exists(path):
            pyinstaller_exe = path
            break

    if not pyinstaller_exe:
        print("Error: PyInstaller not found. Please install it with: pip install pyinstaller")
        return False

    print(f"Using PyInstaller: {pyinstaller_exe}")

    # PyInstaller command for PySide6 application
    cmd = [
        pyinstaller_exe,
        "--name=PyMeshIt",
        "--windowed",  # No console window
        "--onedir",   # Single executable file
        "--noconfirm", # Overwrite existing build
        "--clean",    # Clean cache and temporary files
        "--noupx",    # Don't use UPX compression (can cause issues)
        "--log-level=WARN", # Reduce log verbosity to focus on warnings/errors
        # Main script
        "main.py",
        # Include data files
        "--add-data=resources;resources",
        "--add-data=Pymeshit;Pymeshit",
        # Hidden imports for PySide6
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=shiboken6",
        # Hidden imports for essential scientific packages only
        "--hidden-import=scipy",
        "--hidden-import=scipy.sparse",
        "--hidden-import=numpy",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.pyplot",
        "--hidden-import=PIL",
        "--hidden-import=pyvista",
        "--hidden-import=tetgen",
        "--hidden-import=triangle",
        # Exclude unnecessary packages to reduce build size
        "--exclude-module=torch",
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build completed successfully!")
        print(result.stdout)

        # Check if exe was created
        exe_path = project_root / "release" / "MeshIt.exe"
        if exe_path.exists():
            print(f"\nExecutable created: {exe_path}")
            print(f"File size: {exe_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("\nWarning: Executable not found in expected location")

    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

    return True

def clean_build():
    """Clean build artifacts"""
    import shutil

    dirs_to_remove = ["build", "MeshIt.spec"]
    for item in dirs_to_remove:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
            print(f"Removed: {item}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean_build()
    else:
        success = build_exe()
        if success:
            #print(" Build completed successfully!")
            print("You can find the executable in the 'release' folder")
        else:
            print("Build failed!")
            sys.exit(1)
