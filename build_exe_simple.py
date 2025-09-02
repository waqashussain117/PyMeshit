#!/usr/bin/env python3
"""
Simple PyInstaller build script for MeshIt
"""

import os
import sys
import subprocess

def main():
    """Build MeshIt executable using PyInstaller"""

    print("Building MeshIt executable...")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")

    # Basic PyInstaller command
    cmd = [
        sys.executable, "-m", "pyinstaller",
        "--name=MeshIt",
        "--windowed",
        "--onefile",
        "--noconfirm",
        "main.py",
        "--add-data=resources;resources",
        "--add-data=Pymeshit;Pymeshit",
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=shiboken6",
        "--hidden-import=scipy",
        "--hidden-import=matplotlib",
        "--hidden-import=pyvista",
        "--hidden-import=tetgen",
        "--hidden-import=triangle",
        "--distpath=release",
        "--workpath=build"
    ]

    if os.path.exists("resources/images/app_logo_small.png"):
        cmd.append("--icon=resources/images/app_logo_small.png")

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build completed successfully!")
        print("STDOUT:", result.stdout)

        # Check for the exe
        exe_path = os.path.join("release", "MeshIt.exe")
        if os.path.exists(exe_path):
            file_size = os.path.getsize(exe_path) / (1024*1024)
            print(f"\n✅ Executable created: {exe_path}")
            print(".2f"        else:
            print("\n⚠️  Warning: Executable not found in expected location")

    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("PyInstaller not found! Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("Please run this script again.")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Build completed successfully!")
        print("You can find the executable in the 'release' folder")
    else:
        print("\n❌ Build failed!")
        sys.exit(1)
