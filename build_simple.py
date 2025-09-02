#!/usr/bin/env python3
"""
Simple PyInstaller build script for MeshIt
This script avoids environment detection issues
"""

import os
import sys
import subprocess
import shutil

def main():
    print("🔧 Simple MeshIt Build Script")
    print("=" * 50)

    # Step 1: Clean cache directories
    print("🧹 Cleaning cache files...")
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(pycache_path)
                    print(f"  ✅ Removed: {pycache_path}")
                except:
                    pass  # Ignore errors

    # Step 2: Try to install PyInstaller if missing
    print("📦 Checking PyInstaller...")
    try:
        import PyInstaller
        print("  ✅ PyInstaller found")
    except ImportError:
        print("  📥 Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("  ✅ PyInstaller installed")

    # Step 3: Clean old build files
    print("🧽 Cleaning old build files...")
    for item in ["build", "release", "MeshIt.spec"]:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                print(f"  ✅ Removed: {item}")
            except:
                pass

    # Step 4: Build the executable
    print("🏗️  Building executable...")
    cmd = [
        sys.executable, "-m", "pyinstaller",
        "--name=MeshIt",
        "--windowed",
        "--onefile",
        "--noconfirm",
        "--clean",
        "main.py",
        # Only include essential data files
        "--add-data=resources/images;resources/images",
        "--add-data=Pymeshit/__init__.py;Pymeshit",
        "--add-data=Pymeshit/intersection_utils.py;Pymeshit",
        "--add-data=Pymeshit/tetra_mesh_utils.py;Pymeshit",
        "--add-data=Pymeshit/triangle_direct.py;Pymeshit",
        "--add-data=Pymeshit/core/__init__.py;Pymeshit/core",
        # Essential hidden imports
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=shiboken6",
        "--hidden-import=scipy",
        "--hidden-import=matplotlib",
        "--hidden-import=pyvista",
        "--hidden-import=tetgen",
        "--hidden-import=triangle",
        "--hidden-import=numpy",
        # Output
        "--distpath=release",
        "--workpath=build",
    ]

    if os.path.exists("resources/images/app_logo_small.png"):
        cmd.append("--icon=resources/images/app_logo_small.png")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build completed successfully!")

        # Check the result
        exe_path = "release/MeshIt.exe"
        if os.path.exists(exe_path):
            file_size = os.path.getsize(exe_path) / (1024*1024)
            print("\n📁 Executable created!")
            print(f"   Location: {exe_path}")
            print(".2f")
            print("   Status: Ready for distribution! 🚀")
            return True
        else:
            print("\n❌ Executable not found in expected location")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        print("Error output:", e.stderr)
        return False
    except Exception as e:
        print(f"\n❌ Build failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS! Your optimized MeshIt executable is ready!")
        print("📂 Location: release/MeshIt.exe")
        print("📊 File size: Should be ~2-3 MB (not 2.7 GB!)")
        print("🚀 Share this file with anyone - no Python installation needed!")
    else:
        print("\n❌ Build failed. Check the error messages above.")
        print("💡 Try running: pip install pyinstaller")
    print("=" * 50)
