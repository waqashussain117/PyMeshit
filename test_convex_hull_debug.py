#!/usr/bin/env python3
"""
Test script to debug convex hull visualization in the tetra mesh tab.
"""

import sys
import os
import numpy as np
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_convex_hull_debug():
    """Test convex hull functionality with debug output."""
    try:
        from PyQt5.QtWidgets import QApplication
        from meshit_workflow_gui import MeshItWorkflowGUI
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create main window
        window = MeshItWorkflowGUI()
        
        # Create some test data to simulate hull computation
        # Simple test: square points
        test_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]  # Interior point
        ])
        
        # Create dataset with hull points computed
        test_dataset = {
            'name': 'Test Surface',
            'points': test_points,
            'visible': True,
            'color': '#FF0000',
            'surface_type': 'UNIT'
        }
        
        # Compute hull for the test dataset
        print("Creating test dataset...")
        window.datasets = [test_dataset]
        window.current_dataset_index = 0
        
        # Compute hull
        print("Computing hull for test dataset...")
        success = window._compute_hull_for_dataset(0)
        
        if success:
            print("Hull computation successful!")
            hull_points = window.datasets[0].get('hull_points')
            print(f"Hull points shape: {hull_points.shape if hasattr(hull_points, 'shape') else 'no shape'}")
            print(f"Hull points: {hull_points}")
            
            # Now test the visualization
            print("\nTesting hull visualization...")
            
            # Initialize the tetra plotter if needed
            if not hasattr(window, 'tetra_plotter') or not window.tetra_plotter:
                print("Initializing tetra plotter...")
                # You might need to setup the plotter properly
                # This is just a test to see the debug output
                
            # Test the hull addition function directly
            try:
                print("Testing _add_convex_hull_to_plotter...")
                window._add_convex_hull_to_plotter(0, hull_points)
                print("_add_convex_hull_to_plotter completed without error")
            except Exception as e:
                print(f"Error in _add_convex_hull_to_plotter: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
            
            # Test the filtered hull addition
            try:
                print("Testing _add_filtered_convex_hulls...")
                window._add_filtered_convex_hulls([0])
                print("_add_filtered_convex_hulls completed without error")
            except Exception as e:
                print(f"Error in _add_filtered_convex_hulls: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                
        else:
            print("Hull computation failed!")
        
        print("Test completed. Check the debug output above.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    print("Testing Convex Hull Debug Functionality")
    print("=" * 50)
    test_convex_hull_debug()
