import os
import sys
import glob
import re

def find_qt_installation():
    """Search for Qt installation on the system."""
    
    # Check the Qt installation directory
    qt_base = r"C:\Qt"
    
    if not os.path.exists(qt_base):
        print(f"ERROR: Qt directory not found at {qt_base}")
        return None
    
    print(f"Searching for Qt installation in {qt_base}...")
    
    # More extensive search through all directories in Qt folder
    for root, dirs, files in os.walk(qt_base):
        # First, try to find QtCore.dll as an indicator of Qt binaries
        for file in files:
            if file.lower() in ('qtcore.dll', 'qt6core.dll', 'qt5core.dll'):
                bin_dir = root
                print(f"Found Qt binary at: {os.path.join(bin_dir, file)}")
                
                # Look for an include directory nearby
                parent_dir = os.path.dirname(bin_dir)
                include_dir = os.path.join(parent_dir, "include")
                
                if os.path.exists(include_dir):
                    if os.path.exists(os.path.join(include_dir, "QtCore")):
                        print(f"✅ FOUND Qt SDK at: {parent_dir}")
                        return parent_dir
        
        # Check if this directory has the Qt include structure
        if "include" in dirs:
            include_path = os.path.join(root, "include")
            if os.path.exists(os.path.join(include_path, "QtCore")):
                print(f"✅ FOUND Qt SDK at: {root}")
                return root
    
    # If we still haven't found it, look for any .lib files that might be Qt related
    qt_lib_pattern = re.compile(r"Qt[56]?Core.*\.lib", re.IGNORECASE)
    
    for root, dirs, files in os.walk(qt_base):
        for file in files:
            if qt_lib_pattern.match(file):
                lib_dir = root
                print(f"Found Qt library at: {os.path.join(lib_dir, file)}")
                
                # Try to determine the Qt installation root from the library path
                # Usually lib is in <qt_root>/lib
                parent_dir = os.path.dirname(lib_dir)
                if os.path.basename(lib_dir).lower() == "lib":
                    qt_root = parent_dir
                else:
                    qt_root = lib_dir
                
                print(f"Potential Qt root directory: {qt_root}")
                return qt_root
    
    print("Could not find Qt development files. You might need to install Qt SDK.")
    return None

def show_directory_structure(path, level=0, max_depth=3):
    """Shows directory structure to help debug"""
    if level > max_depth:
        return
    
    indent = '  ' * level
    print(f"{indent}📁 {os.path.basename(path)}")
    
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                show_directory_structure(item_path, level + 1, max_depth)
            elif level == max_depth:
                print(f"{indent}  📄 {item}")
    except PermissionError:
        print(f"{indent}  ⚠️ Permission denied")

if __name__ == "__main__":
    print("Looking for Qt installation...")
    
    # First, show some directories to help debug
    print("\nQt Base Directory Structure:")
    qt_base = r"C:\Qt"
    if os.path.exists(qt_base):
        show_directory_structure(qt_base, max_depth=2)
    else:
        print("Qt base directory not found")
    
    print("\nSearching for Qt SDK...")
    qt_path = find_qt_installation()
    
    if qt_path:
        print(f"\nQt path: {qt_path}")
        
        # Try to determine include and lib paths
        include_path = os.path.join(qt_path, 'include')
        lib_path = os.path.join(qt_path, 'lib')
        
        # Verify paths exist
        if os.path.exists(include_path):
            print(f"Include path: {include_path}")
            # Show available Qt modules
            qt_modules = [d for d in os.listdir(include_path) if d.startswith("Qt")]
            print(f"Available Qt modules: {', '.join(qt_modules)}")
        else:
            print(f"Include path not found at {include_path}")
        
        if os.path.exists(lib_path):
            print(f"Library path: {lib_path}")
            # Count Qt libs
            qt_libs = [f for f in os.listdir(lib_path) if f.startswith("Qt") and f.endswith(".lib")]
            print(f"Found {len(qt_libs)} Qt libraries")
        else:
            print(f"Library path not found at {lib_path}")
        
        # Try to determine Qt version
        if os.path.exists(include_path) and os.path.exists(os.path.join(include_path, "QtCore")):
            version_header = os.path.join(include_path, "QtCore", "qglobal.h")
            if os.path.exists(version_header):
                with open(version_header, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    qt_version_match = re.search(r'#define QT_VERSION_STR "([0-9\.]+)"', content)
                    if qt_version_match:
                        qt_version = qt_version_match.group(1)
                        print(f"Qt version: {qt_version}")
    else:
        print("\nQt development files not found. Please install Qt SDK.")
        print("You can download it from: https://www.qt.io/download")
        sys.exit(1)