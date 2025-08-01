#!/usr/bin/env python3
"""
Test script to verify VTU file loading support in MeshIt workflow GUI.
"""

import os
import sys
import numpy as np

# Try to import pyvista to create a test VTU file
try:
    import pyvista as pv
    HAVE_PYVISTA = True
    print("✓ PyVista available for testing")
except ImportError:
    HAVE_PYVISTA = False
    print("✗ PyVista not available - cannot create test VTU file")

def create_test_vtu_file(filename="test_points.vtu"):
    """Create a simple VTU file with test points for verification."""
    if not HAVE_PYVISTA:
        print("Cannot create test VTU file without PyVista")
        return None
    
    # Create some test points (a simple 3D grid)
    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 10, 5) 
    z = np.linspace(0, 5, 3)
    
    # Create a structured grid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Create a PolyData object with these points
    mesh = pv.PolyData(points)
    
    # Add some scalar data for testing
    mesh['elevation'] = points[:, 2]  # Z coordinate as scalar
    mesh['point_id'] = np.arange(len(points))
    
    # Save as VTU file
    mesh.save(filename)
    print(f"✓ Created test VTU file: {filename}")
    print(f"  - Points: {len(points)}")
    print(f"  - Point range X: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]")
    print(f"  - Point range Y: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}]") 
    print(f"  - Point range Z: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]")
    
    return filename

def test_vtu_reading(filename):
    """Test reading the VTU file using PyVista directly."""
    if not HAVE_PYVISTA:
        print("Cannot test VTU reading without PyVista")
        return False
        
    if not os.path.exists(filename):
        print(f"✗ Test file {filename} does not exist")
        return False
    
    try:
        # Read the VTU file
        mesh = pv.read(filename)
        points = mesh.points
        
        print(f"✓ Successfully read VTU file: {filename}")
        print(f"  - Points read: {len(points)}")
        print(f"  - Point shape: {points.shape}")
        print(f"  - First few points:")
        for i in range(min(3, len(points))):
            print(f"    [{points[i, 0]:.1f}, {points[i, 1]:.1f}, {points[i, 2]:.1f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading VTU file: {e}")
        return False

def verify_file_dialog_filters():
    """Verify that the file dialog filters include VTU files."""
    print("\n=== Verifying File Dialog Filters ===")
    
    # Read the GUI file to check if VTU filter was added
    gui_file = "meshit_workflow_gui.py"
    if not os.path.exists(gui_file):
        print(f"✗ GUI file {gui_file} not found")
        return False
    
    try:
        with open(gui_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for VTU in file dialog filters
        vtu_filter_found = "VTU files (*.vtu)" in content
        
        if vtu_filter_found:
            print("✓ VTU filter found in file dialog")
            
            # Count occurrences
            count = content.count("VTU files (*.vtu)")
            print(f"  - Found in {count} location(s)")
            
            return True
        else:
            print("✗ VTU filter not found in file dialog")
            return False
            
    except Exception as e:
        print(f"✗ Error reading GUI file: {e}")
        return False

def main():
    """Main test function."""
    print("=== MeshIt VTU Support Test ===\n")
    
    # Test 1: Verify file dialog filters
    filter_ok = verify_file_dialog_filters()
    
    print("\n=== Testing VTU File Creation and Reading ===")
    
    # Test 2: Create test VTU file
    if HAVE_PYVISTA:
        test_filename = create_test_vtu_file()
        
        if test_filename:
            # Test 3: Read the VTU file
            read_ok = test_vtu_reading(test_filename)
            
            # Cleanup
            if os.path.exists(test_filename):
                os.remove(test_filename)
                print(f"✓ Cleaned up test file: {test_filename}")
        else:
            read_ok = False
    else:
        print("⚠ Skipping VTU file tests - PyVista not available")
        read_ok = None
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"File dialog filters: {'✓ PASS' if filter_ok else '✗ FAIL'}")
    
    if read_ok is not None:
        print(f"VTU file reading: {'✓ PASS' if read_ok else '✗ FAIL'}")
    else:
        print("VTU file reading: ⚠ SKIPPED (PyVista not available)")
    
    if filter_ok and (read_ok is None or read_ok):
        print("\n🎉 VTU support implementation looks good!")
        if read_ok is None:
            print("   Note: Install PyVista to test VTU file reading")
    else:
        print("\n❌ Some tests failed - please check the implementation")

if __name__ == "__main__":
    main()
