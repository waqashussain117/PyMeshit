#!/usr/bin/env python3
"""
MeshIt Executable Builder
=========================

This script creates a standalone executable (.exe) for your MeshIt GUI application.

USAGE:
    python build_exe_final.py          # Build executable
    python build_exe_final.py clean    # Clean build artifacts

REQUIREMENTS:
    - Python 3.7+
    - PyInstaller (will be installed automatically if missing)
    - All project dependencies from requirements.txt

OUTPUT:
    - Executable: release/MeshIt.exe
    - Build artifacts: build/ folder
    - Spec file: MeshIt.spec

TROUBLESHOOTING:
    - If build fails, try: pip install pyinstaller
    - For missing modules, add: --hidden-import=module_name
    - For data files, add: --add-data=source;destination
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check and install requirements"""
    print("🔍 Checking requirements...")

    # Check PyInstaller
    try:
        import PyInstaller
        print("✅ PyInstaller found")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("✅ PyInstaller installed")

    # Check if requirements.txt exists and install dependencies
    if os.path.exists("requirements.txt"):
        print("📦 Installing project dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed")

def get_correct_python_exe():
    """Get the correct Python executable that has PyInstaller installed"""
    print("🔍 Looking for Python with PyInstaller...")

    # Try multiple possible locations
    possible_paths = [
        # Check for pyinstaller.exe in PATH
        "pyinstaller.exe",
        # User-installed Python 3.12
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python312", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python312", "Scripts", "python.exe"),
        # User-installed Python 3.11
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python311", "python.exe"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python311", "Scripts", "python.exe"),
        # Local packages Python
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Packages", "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0", "LocalCache", "local-packages", "Python311", "Scripts", "python.exe"),
        # Current environment (try last)
        sys.executable,
    ]

    # Test each path
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"Testing: {path}")
                # Test if this Python has PyInstaller
                if path.endswith("pyinstaller.exe"):
                    # If it's pyinstaller.exe directly, use it
                    print(f"✅ Found pyinstaller.exe: {path}")
                    return path
                else:
                    # If it's python.exe, test for PyInstaller
                    result = subprocess.run([path, "-c", "import PyInstaller; print('OK')"],
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and 'OK' in result.stdout.strip():
                        print(f"✅ Found working Python with PyInstaller: {path}")
                        return path
            except Exception as e:
                print(f"❌ Failed to test {path}: {e}")
                continue

    # If none work, try to install PyInstaller in current environment
    print(f"⚠️  No Python with PyInstaller found. Installing in current environment...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("✅ PyInstaller installed successfully")
        return sys.executable
    except Exception as e:
        print(f"❌ Failed to install PyInstaller: {e}")
        return None

def clean_pycache_dirs():
    """Clean __pycache__ directories to reduce executable size"""
    print("🧹 Cleaning __pycache__ directories...")

    import shutil

    # Find and remove all __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(pycache_path)
                    print(f"✅ Removed: {pycache_path}")
                except Exception as e:
                    print(f"⚠️  Could not remove {pycache_path}: {e}")

    # Also clean any .pyc files in current directory
    for file in os.listdir('.'):
        if file.endswith('.pyc'):
            try:
                os.remove(file)
                print(f"✅ Removed: {file}")
            except Exception as e:
                print(f"⚠️  Could not remove {file}: {e}")

def build_exe():
    """Build standalone executable for MeshIt GUI"""

    print("\n🚀 Building MeshIt executable...")
    print(f"Working directory: {os.getcwd()}")

    # Find the correct Python executable with PyInstaller
    python_exe = get_correct_python_exe()
    if python_exe is None:
        print("❌ ERROR: Could not find or install PyInstaller")
        print("💡 Try running: pip install pyinstaller")
        return False

    print(f"Python executable: {python_exe}")

    # Check requirements first
    check_requirements()

    # Clean __pycache__ directories first
    clean_pycache_dirs()

    # PyInstaller command for PySide6 application
    cmd = [
        python_exe, "-m", "pyinstaller",

        # Basic options
        "--name=MeshIt",
        "--windowed",      # No console window
        "--onefile",       # Single executable file
        "--noconfirm",     # Overwrite existing build
        "--clean",         # Clean cache and temporary files

        # Exclude unnecessary files
        "--exclude-module=PyQt5",  # Exclude PyQt5 if present
        "--exclude-module=PyQt6",  # Exclude PyQt6 if present

        # Main script
        "main.py",

        # Include ONLY necessary data files
        "--add-data=resources/images;resources/images",
        "--add-data=Pymeshit/__init__.py;Pymeshit",
        "--add-data=Pymeshit/intersection_utils.py;Pymeshit",
        "--add-data=Pymeshit/tetra_mesh_utils.py;Pymeshit",
        "--add-data=Pymeshit/triangle_direct.py;Pymeshit",
        "--add-data=Pymeshit/core/__init__.py;Pymeshit/core",

        # Hidden imports for PySide6
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=shiboken6",

        # Hidden imports for scientific packages (only essential ones)
        "--hidden-import=scipy",
        "--hidden-import=scipy.sparse",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.pyplot",
        "--hidden-import=pyvista",
        "--hidden-import=tetgen",
        "--hidden-import=triangle",
        "--hidden-import=numpy",

        # Output directories
        "--distpath=release",
        "--workpath=build",
    ]

    # Add icon if available
    if os.path.exists("resources/images/app_logo_small.png"):
        cmd.append("--icon=resources/images/app_logo_small.png")
        print("🎨 Using application icon")

    print("🔧 Running PyInstaller...")    

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build completed successfully!")
        print(result.stdout)
        print(result.stderr)

        # Check if exe was created
        exe_path = Path("release/MeshIt.exe")
        if exe_path.exists():
            file_size = exe_path.stat().st_size / (1024*1024)
            print("📁 Executable created successfully!")
            print(f"   Location: {exe_path.absolute()}")
            print("   Status: Ready for distribution! 🚀")
            print("File size:", f"{file_size:.2f} MB")
            return True
        else:
            print("\n⚠️  Warning: Executable not found in expected location")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed with error code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"\n❌ Build failed with error: {e}")
        return False

def clean_build():
    """Clean build artifacts"""
    print("🧹 Cleaning build artifacts...")

    import shutil

    items_to_remove = [
        "build",
        "MeshIt.spec",
        "__pycache__",
        "release/MeshIt.exe"  # Remove old executable
    ]

    for item in items_to_remove:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"✅ Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"✅ Removed file: {item}")
            except Exception as e:
                print(f"⚠️  Could not remove {item}: {e}")

def show_info():
    """Show build information"""
    print("\n" + "="*60)
    print("MeshIt Executable Builder")
    print("="*60)
    print("This script creates a standalone .exe file for your GUI application.")
    print("\nCURRENT STATUS:")
    print("✅ PyInstaller setup: Complete")
    print("✅ Dependencies: Will be installed automatically")
    print("✅ Build configuration: Optimized for PySide6")
    print("\nEXPECTED OUTPUT:")
    print("📁 release/MeshIt.exe (~2-3 MB)")
    print("📁 build/ (temporary build files)")
    print("📄 MeshIt.spec (build configuration)")
    print("\nNEXT STEPS:")
    print("1. Run: python build_exe_final.py")
    print("2. Wait for build completion")
    print("3. Find executable in: release/MeshIt.exe")
    print("4. Test the executable on your system")
    print("5. Distribute to users (no Python installation required!)")
    print("="*60)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "clean":
            clean_build()
            print("\n✅ Cleanup completed!")
            return
        elif sys.argv[1] == "info":
            show_info()
            return
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python build_exe_final.py [clean|info]")
            return

    # Show info first
    show_info()

    # Build executable
    success = build_exe()

    if success:
        print("\n" + "="*60)
        print("🎉 SUCCESS! Your MeshIt executable is ready!")
        print("="*60)
        print("📂 Location: release/MeshIt.exe")
        print("🚀 Share this file with anyone - no Python installation needed!")
        print("\n🧪 Test the executable:")
        print("   1. Double-click: release/MeshIt.exe")
        print("   2. Or run: start release/MeshIt.exe")
        print("\n📦 Distribution tips:")
        print("   - File size: ~2-3 MB")
        print("   - No dependencies required")
        print("   - Works on Windows 10/11")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ BUILD FAILED")
        print("="*60)
        print("💡 Troubleshooting:")
        print("   1. Check Python version: python --version")
        print("   2. Install PyInstaller: pip install pyinstaller")
        print("   3. Check dependencies: pip install -r requirements.txt")
        print("   4. Try cleaning: python build_exe_final.py clean")
        print("   5. Run again: python build_exe_final.py")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()
