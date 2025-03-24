"""
MeshIt Qt GUI

This application provides a graphical interface for the MeshIt workflow
using PyQt5 with embedded PyVista 3D visualization.
"""

import sys
import os
import numpy as np
import logging
import time

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QFileDialog,
    QMessageBox, QSplitter, QFrame, QStatusBar, QDialog
)
from PyQt5.QtCore import Qt, QSize

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MeshIt-Qt")

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvista.qt import QtInteractor
    HAVE_PYVISTA = True
    logger.info("Successfully imported PyVista for 3D visualization")
except ImportError:
    HAVE_PYVISTA = False
    logger.warning("PyVista not available. 3D visualization disabled.")

# Import the DirectTriangleWrapper
try:
    from meshit.triangle_direct import DirectTriangleWrapper
    HAVE_DIRECT_WRAPPER = True
    logger.info("Successfully imported DirectTriangleWrapper")
except ImportError as e:
    logger.error(f"Failed to import DirectTriangleWrapper: {e}")
    HAVE_DIRECT_WRAPPER = False

class MeshItQtGUI(QMainWindow):
    """
    Main window for the MeshIt Qt GUI application
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeshIt Qt GUI")
        self.setMinimumSize(1200, 800)
        
        # Initialize data containers
        self.points = None
        self.hull_points = None
        self.segments = None
        self.triangulation_result = None
        
        # Initialize PyVista 3D visualization
        self.view_3d_enabled = HAVE_PYVISTA
        self.current_plotter = None
        self.height_factor = 0.2
        
        # Initialize datasets for multi-surface viewing
        self.datasets = {}
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the main UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs for each workflow step
        self.load_tab = QWidget()
        self.hull_tab = QWidget()
        self.segment_tab = QWidget()
        self.triangulation_tab = QWidget()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.load_tab, "1. Load Data")
        self.tab_widget.addTab(self.hull_tab, "2. Convex Hull")
        self.tab_widget.addTab(self.segment_tab, "3. Segmentation")
        self.tab_widget.addTab(self.triangulation_tab, "4. Triangulation")
        
        # Setup individual tabs
        self.setup_load_tab()
        self.setup_hull_tab()
        self.setup_segment_tab()
        self.setup_triangulation_tab()
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
    
    def setup_load_tab(self):
        """Set up the load data tab"""
        # Main layout
        layout = QHBoxLayout(self.load_tab)
        
        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel (controls)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)
        
        # File loading group
        file_group = QGroupBox("Data Import")
        file_layout = QVBoxLayout(file_group)
        
        # Load file button
        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)
        
        # Generate test data button
        test_btn = QPushButton("Generate Test Data")
        test_btn.clicked.connect(self.generate_test_data)
        file_layout.addWidget(test_btn)
        
        control_layout.addWidget(file_group)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        
        # Point count
        self.points_label = QLabel("Points: 0")
        stats_layout.addRow(self.points_label)
        
        # Bounds
        self.bounds_label = QLabel("Bounds: N/A")
        stats_layout.addRow(self.bounds_label)
        
        control_layout.addWidget(stats_group)
        
        # Add stretch to push controls to the top
        control_layout.addStretch()
        
        # Right panel (visualization)
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        splitter.addWidget(viz_panel)
        
        # Visualization title
        viz_title = QLabel("Point Cloud Visualization")
        viz_title.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(viz_title)
        
        # Visualization area
        self.load_viz_container = QWidget()
        self.load_viz_layout = QVBoxLayout(self.load_viz_container)
        viz_layout.addWidget(self.load_viz_container)
        
        # Set initial text
        if HAVE_PYVISTA:
            placeholder_text = "Load data to visualize points in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization."
        
        placeholder = QLabel(placeholder_text)
        placeholder.setAlignment(Qt.AlignCenter)
        self.load_viz_layout.addWidget(placeholder)
        
        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([300, 700])
    
    def setup_hull_tab(self):
        """Set up the convex hull tab"""
        # Main layout
        layout = QHBoxLayout(self.hull_tab)
        
        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel (controls)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)
        
        # Hull controls group
        hull_group = QGroupBox("Hull Controls")
        hull_layout = QVBoxLayout(hull_group)
        
        # Compute hull button
        compute_btn = QPushButton("Compute Convex Hull")
        compute_btn.clicked.connect(self.compute_hull)
        hull_layout.addWidget(compute_btn)
        
        control_layout.addWidget(hull_group)
        
        # Statistics group
        stats_group = QGroupBox("Hull Statistics")
        stats_layout = QFormLayout(stats_group)
        
        # Hull points count
        self.hull_points_label = QLabel("Hull vertices: 0")
        stats_layout.addRow(self.hull_points_label)
        
        # Hull area
        self.hull_area_label = QLabel("Hull area: 0.0")
        stats_layout.addRow(self.hull_area_label)
        
        control_layout.addWidget(stats_group)
        
        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(0))
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        nav_layout.addWidget(next_btn)
        
        control_layout.addWidget(nav_group)
        
        # Add stretch to push controls to the top
        control_layout.addStretch()
        
        # Right panel (visualization)
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        splitter.addWidget(viz_panel)
        
        # Visualization title
        viz_title = QLabel("Convex Hull Visualization")
        viz_title.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(viz_title)
        
        # Visualization area 
        self.hull_viz_container = QWidget()
        self.hull_viz_layout = QVBoxLayout(self.hull_viz_container)
        viz_layout.addWidget(self.hull_viz_container)
        
        # Set initial text
        if HAVE_PYVISTA:
            placeholder_text = "Compute convex hull to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization."
        
        placeholder = QLabel(placeholder_text)
        placeholder.setAlignment(Qt.AlignCenter)
        self.hull_viz_layout.addWidget(placeholder)
        
        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([300, 700])
    
    def setup_segment_tab(self):
        """Set up the segmentation tab"""
        # Main layout
        layout = QHBoxLayout(self.segment_tab)
        
        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel (controls)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)
        
        # Segmentation controls group
        segment_group = QGroupBox("Segmentation Controls")
        segment_layout = QFormLayout(segment_group)
        
        # Segment length control
        self.segment_length_spin = QDoubleSpinBox()
        self.segment_length_spin.setRange(0.1, 100.0)
        self.segment_length_spin.setValue(1.0)
        self.segment_length_spin.setSingleStep(0.1)
        segment_layout.addRow("Segment Length:", self.segment_length_spin)
        
        # Segment density control
        self.segment_density_spin = QDoubleSpinBox()
        self.segment_density_spin.setRange(0.5, 2.0)
        self.segment_density_spin.setValue(1.0)
        self.segment_density_spin.setSingleStep(0.1)
        segment_layout.addRow("Density:", self.segment_density_spin)
        
        # Compute segments button
        compute_btn = QPushButton("Compute Segmentation")
        compute_btn.clicked.connect(self.compute_segments)
        segment_layout.addRow(compute_btn)
        
        control_layout.addWidget(segment_group)
        
        # Statistics group
        stats_group = QGroupBox("Segment Statistics")
        stats_layout = QFormLayout(stats_group)
        
        # Number of segments
        self.num_segments_label = QLabel("Segments: 0")
        stats_layout.addRow(self.num_segments_label)
        
        # Average segment length
        self.avg_segment_length_label = QLabel("Avg length: 0.0")
        stats_layout.addRow(self.avg_segment_length_label)
        
        control_layout.addWidget(stats_group)
        
        # Navigation buttons
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(3))
        nav_layout.addWidget(next_btn)
        
        control_layout.addWidget(nav_group)
        
        # Add stretch to push controls to the top
        control_layout.addStretch()
        
        # Right panel (visualization)
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        splitter.addWidget(viz_panel)
        
        # Visualization title
        viz_title = QLabel("Segmentation Visualization")
        viz_title.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(viz_title)
        
        # Visualization area
        self.segment_viz_container = QWidget()
        self.segment_viz_layout = QVBoxLayout(self.segment_viz_container)
        viz_layout.addWidget(self.segment_viz_container)
        
        # Set initial text
        if HAVE_PYVISTA:
            placeholder_text = "Compute segmentation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization."
        
        placeholder = QLabel(placeholder_text)
        placeholder.setAlignment(Qt.AlignCenter)
        self.segment_viz_layout.addWidget(placeholder)
        
        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([300, 700])
    
    def setup_triangulation_tab(self):
        """Set up the triangulation tab"""
        # Main layout
        layout = QHBoxLayout(self.triangulation_tab)
        
        # Create a splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel (controls)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)
        
        # Triangulation controls group
        tri_group = QGroupBox("Triangulation Controls")
        tri_layout = QFormLayout(tri_group)
        
        # Mesh density controls
        self.base_size_factor_spin = QDoubleSpinBox()
        self.base_size_factor_spin.setRange(5.0, 30.0)
        self.base_size_factor_spin.setValue(15.0)
        self.base_size_factor_spin.setSingleStep(1.0)
        tri_layout.addRow("Mesh Density:", self.base_size_factor_spin)
        
        # Quality settings subgroup
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QFormLayout(quality_group)
        
        # Gradient
        self.gradient_spin = QDoubleSpinBox()
        self.gradient_spin.setRange(1.0, 3.0)
        self.gradient_spin.setValue(1.0)
        self.gradient_spin.setSingleStep(0.1)
        quality_layout.addRow("Gradient:", self.gradient_spin)
        
        # Min angle
        self.min_angle_spin = QDoubleSpinBox()
        self.min_angle_spin.setRange(10.0, 30.0)
        self.min_angle_spin.setValue(25.0)
        self.min_angle_spin.setSingleStep(1.0)
        quality_layout.addRow("Min Angle:", self.min_angle_spin)
        
        # Uniform triangulation
        self.uniform_check = QCheckBox()
        self.uniform_check.setChecked(True)
        quality_layout.addRow("Uniform:", self.uniform_check)
        
        tri_layout.addRow(quality_group)
        
        # Feature points subgroup
        feature_group = QGroupBox("Feature Points")
        feature_layout = QFormLayout(feature_group)
        
        # Use features
        self.use_feature_points_check = QCheckBox()
        self.use_feature_points_check.setChecked(False)
        feature_layout.addRow("Use Features:", self.use_feature_points_check)
        
        # Number of features
        self.num_features_spin = QSpinBox()
        self.num_features_spin.setRange(1, 10)
        self.num_features_spin.setValue(3)
        feature_layout.addRow("Count:", self.num_features_spin)
        
        # Feature size
        self.feature_size_spin = QDoubleSpinBox()
        self.feature_size_spin.setRange(0.1, 3.0)
        self.feature_size_spin.setValue(1.0)
        self.feature_size_spin.setSingleStep(0.1)
        feature_layout.addRow("Size:", self.feature_size_spin)
        
        tri_layout.addRow(feature_group)
        
        # 3D visualization settings
        if HAVE_PYVISTA:
            viz3d_group = QGroupBox("3D Settings")
            viz3d_layout = QFormLayout(viz3d_group)
            
            # Height scale
            self.height_factor_spin = QDoubleSpinBox()
            self.height_factor_spin.setRange(0.0, 1.0)
            self.height_factor_spin.setValue(0.2)
            self.height_factor_spin.setSingleStep(0.1)
            self.height_factor_spin.valueChanged.connect(self.update_height_factor)
            viz3d_layout.addRow("Height Scale:", self.height_factor_spin)
            
            tri_layout.addRow(viz3d_group)
            
            # Multi-surface controls
            multi_surface_group = QGroupBox("Multi-Surface Display")
            multi_surface_layout = QVBoxLayout(multi_surface_group)
            
            # Add dataset to view
            add_btn = QPushButton("Add Current to 3D View")
            add_btn.clicked.connect(self.add_current_to_3d_view)
            multi_surface_layout.addWidget(add_btn)
            
            # Clear datasets
            clear_btn = QPushButton("Clear All Surfaces")
            clear_btn.clicked.connect(self.clear_all_surfaces)
            multi_surface_layout.addWidget(clear_btn)
            
            tri_layout.addRow(multi_surface_group)
        
        # Run triangulation button
        run_btn = QPushButton("Run Triangulation")
        run_btn.clicked.connect(self.run_triangulation)
        tri_layout.addRow(run_btn)
        
        control_layout.addWidget(tri_group)
        
        # Statistics group
        stats_group = QGroupBox("Triangulation Statistics")
        stats_layout = QFormLayout(stats_group)
        
        # Number of triangles
        self.num_triangles_label = QLabel("Triangles: 0")
        stats_layout.addRow(self.num_triangles_label)
        
        # Number of vertices
        self.num_vertices_label = QLabel("Vertices: 0")
        stats_layout.addRow(self.num_vertices_label)
        
        # Mean edge length
        self.mean_edge_label = QLabel("Mean edge: 0.0")
        stats_layout.addRow(self.mean_edge_label)
        
        # Edge uniformity
        self.uniformity_label = QLabel("Uniformity: 0.0")
        stats_layout.addRow(self.uniformity_label)
        
        # Surface info
        self.surfaces_label = QLabel("Surfaces: 0")
        stats_layout.addRow(self.surfaces_label)
        
        control_layout.addWidget(stats_group)
        
        # Export button
        export_btn = QPushButton("Export Results...")
        export_btn.clicked.connect(self.export_results)
        control_layout.addWidget(export_btn)
        
        # Navigation buttons
        nav_group = QWidget()
        nav_layout = QHBoxLayout(nav_group)
        
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        nav_layout.addWidget(prev_btn)
        
        control_layout.addWidget(nav_group)
        
        # Add stretch to push controls to the top
        control_layout.addStretch()
        
        # Right panel (visualization)
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        splitter.addWidget(viz_panel)
        
        # Visualization title
        viz_title = QLabel("3D Visualization")
        viz_title.setAlignment(Qt.AlignCenter)
        viz_layout.addWidget(viz_title)
        
        # Visualization area with integrated PyVista plotter
        self.tri_viz_container = QWidget()
        self.tri_viz_layout = QVBoxLayout(self.tri_viz_container)
        viz_layout.addWidget(self.tri_viz_container)
        
        # Create PyVista plotter if available
        if HAVE_PYVISTA:
            try:
                frame = QFrame()
                layout = QVBoxLayout(frame)
                
                # Create plotter
                self.plotter = QtInteractor(frame)
                layout.addWidget(self.plotter.interactor)
                
                # Set background color
                self.plotter.set_background("#383F51")  # Dark blue background
                
                # Add placeholder text
                self.plotter.add_text("Run triangulation to visualize in 3D", 
                                       position='upper_edge', font_size=14, color='white')
                
                # Show axes
                self.plotter.show_axes()
                
                self.tri_viz_layout.addWidget(frame)
                
                # Store reference to the main plotter
                self.main_plotter = self.plotter
            except Exception as e:
                logger.error(f"Error creating PyVista plotter: {e}")
                placeholder = QLabel("Error creating PyVista visualization")
                placeholder.setAlignment(Qt.AlignCenter)
                self.tri_viz_layout.addWidget(placeholder)
        else:
            placeholder = QLabel("PyVista not available. Install PyVista for 3D visualization.")
            placeholder.setAlignment(Qt.AlignCenter)
            self.tri_viz_layout.addWidget(placeholder)
        
        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([300, 700])
    
    def load_file(self):
        """Load data from a file"""
        self.statusBar.showMessage("Loading file...")
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a point data file",
            "",
            "Text files (*.txt);;Data files (*.dat);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            self.statusBar.showMessage("File loading canceled")
            return
        
        # Update status with file path
        self.statusBar.showMessage(f"Loading file: {os.path.basename(file_path)}...")
        
        try:
            # Try to read the file
            points = self._read_point_file(file_path)
            
            if points is not None and len(points) > 0:
                self.points = points
                
                # Visualize the loaded points
                self._plot_points(points)
                
                self.statusBar.showMessage(f"Successfully loaded {len(points)} points")
                
                # Clear any previous results from later steps
                self.hull_points = None
                self.segments = None
                self.triangulation_result = None
                
                # TODO: Update visualization in other tabs when implemented
                
                # Update the points count in the UI
                self.points_label.setText(f"Points: {len(points)}")
                
                # Calculate and display bounds
                min_x, min_y = np.min(points, axis=0)
                max_x, max_y = np.max(points, axis=0)
                self.bounds_label.setText(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}]")
                
                # Auto-switch to the load tab
                self.tab_widget.setCurrentIndex(0)
            else:
                self.statusBar.showMessage("Error: No valid points found in file")
                QMessageBox.warning(self, "Error", "No valid points found in file")
        
        except Exception as e:
            self.statusBar.showMessage(f"Error loading file: {str(e)}")
            logger.error(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def _read_point_file(self, file_path):
        """Read points from a file
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            numpy array of points or None if an error occurred
        """
        try:
            # Try different formats based on file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                # CSV format - try comma, tab, and space separators
                try:
                    points = np.loadtxt(file_path, delimiter=',')
                except:
                    try:
                        points = np.loadtxt(file_path, delimiter='\t')
                    except:
                        points = np.loadtxt(file_path, delimiter=' ')
            else:
                # Default format - space or tab separated
                try:
                    points = np.loadtxt(file_path)
                except:
                    try:
                        points = np.loadtxt(file_path, delimiter='\t')
                    except:
                        points = np.loadtxt(file_path, delimiter=',')
            
            # Check if the file has 2D or 3D points
            if points.shape[1] < 2:
                raise ValueError("File must contain at least 2D points (x, y)")
            
            # Use only first 2 dimensions for now
            return points[:, 0:2]
        
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error reading file: {str(e)}")
            return None

    def _plot_points(self, points):
        """Visualize the points using PyVista or plain Qt
        
        Args:
            points: Points to visualize (Nx2 array)
        """
        if points is None or len(points) == 0:
            return
        
        # Clear existing visualization in load tab
        for i in reversed(range(self.load_viz_layout.count())): 
            self.load_viz_layout.itemAt(i).widget().setParent(None)
        
        # If PyVista is available, use 3D visualization
        if HAVE_PYVISTA and hasattr(self, 'plotter'):
            # Create points visualization for load tab
            try:
                # Create a new plotter for the points
                frame = QFrame()
                layout = QVBoxLayout(frame)
                
                points_plotter = QtInteractor(frame)
                layout.addWidget(points_plotter.interactor)
                
                # Set background color
                points_plotter.set_background("#383F51")  # Dark blue background
                
                # Convert 2D points to 3D
                points_3d = np.zeros((len(points), 3))
                points_3d[:, 0] = points[:, 0]
                points_3d[:, 1] = points[:, 1]
                
                # Create point cloud
                point_cloud = pv.PolyData(points_3d)
                
                # Add points to plotter
                points_plotter.add_points(point_cloud, 
                                          color='blue', 
                                          point_size=5, 
                                          render_points_as_spheres=True)
                
                # Add text and axes
                points_plotter.add_text(f"Point Cloud: {len(points)} points", 
                                       position='upper_edge', font_size=12, color='white')
                points_plotter.show_axes()
                points_plotter.reset_camera()
                
                # Add to layout
                self.load_viz_layout.addWidget(frame)
                
                # Store reference
                self.points_plotter = points_plotter
                
            except Exception as e:
                logger.error(f"Error creating PyVista points visualization: {e}")
                # Fall back to basic visualization if PyVista fails
                self._create_basic_points_visualization(points)
        else:
            # Create basic points visualization using Qt
            self._create_basic_points_visualization(points)

    def _create_basic_points_visualization(self, points):
        """Create a basic visualization of points using Qt
        
        Args:
            points: Points to visualize (Nx2 array)
        """
        # Create a QLabel with text info
        info_label = QLabel(f"Loaded {len(points)} points (PyVista not available for 3D view)")
        info_label.setAlignment(Qt.AlignCenter)
        self.load_viz_layout.addWidget(info_label)
        
        # TODO: Add actual 2D visualization with Qt widgets
        # For now, just display a placeholder
        placeholder = QLabel("2D visualization will be implemented later")
        placeholder.setAlignment(Qt.AlignCenter)
        self.load_viz_layout.addWidget(placeholder)
    
    def generate_test_data(self):
        """Generate test data by showing a dialog to select the type and parameters"""
        self.statusBar.showMessage("Generating test data...")
        
        # Create a dialog for test data options
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Test Data")
        dialog.setMinimumWidth(350)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Data type group
        type_group = QGroupBox("Data Type")
        type_layout = QVBoxLayout(type_group)
        
        # Radio buttons for data types
        data_type_random = QCheckBox("Random Points")
        data_type_random.setChecked(True)
        data_type_circle = QCheckBox("Circle")
        data_type_square = QCheckBox("Square")
        data_type_complex = QCheckBox("Complex Shape")
        
        # Add radio buttons to type group
        type_layout.addWidget(data_type_random)
        type_layout.addWidget(data_type_circle)
        type_layout.addWidget(data_type_square)
        type_layout.addWidget(data_type_complex)
        
        # Add type group to main layout
        layout.addWidget(type_group)
        
        # Number of points
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)
        
        num_points_spin = QSpinBox()
        num_points_spin.setRange(10, 1000)
        num_points_spin.setValue(100)
        num_points_spin.setSingleStep(10)
        params_layout.addRow("Number of points:", num_points_spin)
        
        layout.addWidget(params_group)
        
        # Button group
        button_box = QHBoxLayout()
        
        generate_btn = QPushButton("Generate")
        cancel_btn = QPushButton("Cancel")
        
        button_box.addWidget(generate_btn)
        button_box.addWidget(cancel_btn)
        
        layout.addLayout(button_box)
        
        # Connect buttons
        generate_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            points = None
            data_types = []
            
            if data_type_random.isChecked():
                data_types.append("random")
            if data_type_circle.isChecked():
                data_types.append("circle")
            if data_type_square.isChecked():
                data_types.append("square")
            if data_type_complex.isChecked():
                data_types.append("complex")
            
            num_points = num_points_spin.value()
            
            # Generate points for selected types
            if not data_types:
                QMessageBox.warning(self, "Error", "Please select at least one data type")
                return
            
            try:
                # Create combined point sets
                all_points = []
                points_per_type = num_points // len(data_types)
                
                for data_type in data_types:
                    if data_type == "random":
                        points = self._generate_random_points(points_per_type)
                    elif data_type == "circle":
                        points = self._generate_circle_points(points_per_type)
                    elif data_type == "square":
                        points = self._generate_square_points(points_per_type)
                    elif data_type == "complex":
                        points = self._generate_complex_shape(points_per_type)
                    
                    if points is not None:
                        all_points.append(points)
                
                # Combine points
                combined_points = np.vstack(all_points)
                
                # Store and visualize
                self.points = combined_points
                self._plot_points(combined_points)
                
                # Update UI
                self.points_label.setText(f"Points: {len(combined_points)}")
                
                # Calculate and display bounds
                min_x, min_y = np.min(combined_points, axis=0)
                max_x, max_y = np.max(combined_points, axis=0)
                self.bounds_label.setText(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}]")
                
                # Clear any previous results from later steps
                self.hull_points = None
                self.segments = None
                self.triangulation_result = None
                
                # Update status
                self.statusBar.showMessage(f"Generated {len(combined_points)} points")
                
            except Exception as e:
                self.statusBar.showMessage(f"Error generating test data: {str(e)}")
                logger.error(f"Error generating test data: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error generating test data: {str(e)}")
        else:
            self.statusBar.showMessage("Test data generation canceled")

    def _generate_random_points(self, num_points=100):
        """Generate random points
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            numpy array of points
        """
        # Generate random points in a square
        points = np.random.rand(num_points, 2) * 20 - 10  # Range: -10 to 10
        return points

    def _generate_circle_points(self, num_points=100):
        """Generate points in a circle
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            numpy array of points
        """
        # Generate random points in a circle
        r = 10.0 * np.sqrt(np.random.rand(num_points))  # Radius: 0 to 10
        theta = np.random.rand(num_points) * 2 * np.pi  # Angle: 0 to 2pi
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.column_stack((x, y))

    def _generate_square_points(self, num_points=100):
        """Generate points in a square
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            numpy array of points
        """
        # Generate random points in a square with some clustering
        points = []
        
        # Add corner points to ensure a square shape
        corners = [[-10, -10], [10, -10], [10, 10], [-10, 10]]
        points.extend(corners)
        
        # Add more random points
        remaining = num_points - len(corners)
        if remaining > 0:
            random_points = np.random.rand(remaining, 2) * 20 - 10  # Range: -10 to 10
            points.extend(random_points)
        
        return np.array(points)

    def _generate_complex_shape(self, num_points=100):
        """Generate complex shape with internal features
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            numpy array of points
        """
        points = []
        
        # Generate outer boundary (polygon)
        num_vertices = 8
        radius = 10.0
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        
        # Add some randomness to the radius
        radii = radius * (0.8 + 0.4 * np.random.rand(num_vertices))
        
        # Create polygon vertices
        for i in range(num_vertices):
            x = radii[i] * np.cos(angles[i])
            y = radii[i] * np.sin(angles[i])
            points.append([x, y])
        
        # Add interior points with some clustering
        remaining = num_points - num_vertices
        
        if remaining > 0:
            # Create clusters
            num_clusters = 3
            cluster_centers = []
            
            for _ in range(num_clusters):
                # Random position within the shape (closer to center)
                cx = (np.random.rand() * 10 - 5)
                cy = (np.random.rand() * 10 - 5)
                cluster_centers.append([cx, cy])
            
            # Distribute points among clusters
            for i in range(remaining):
                # Pick a random cluster
                cluster = np.random.randint(0, num_clusters)
                center = cluster_centers[cluster]
                
                # Generate point with noise around the cluster center
                noise_scale = 2.0 + np.random.rand() * 2.0
                x = center[0] + np.random.randn() * noise_scale
                y = center[1] + np.random.randn() * noise_scale
                
                # Limit to within the approximate boundary
                dist = np.sqrt(x*x + y*y)
                if dist > radius:
                    scale = (radius * 0.9) / dist
                    x *= scale
                    y *= scale
                
                points.append([x, y])
        
        return np.array(points)

    def compute_hull(self):
        """Compute the convex hull of the points"""
        self.statusBar.showMessage("Computing convex hull...")
        
        # Check if we have points
        if self.points is None or len(self.points) < 3:
            self.statusBar.showMessage("Error: Need at least 3 points to compute hull")
            QMessageBox.warning(self, "Error", "Need at least 3 points to compute hull")
            return
        
        try:
            # Compute convex hull using scipy
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.points)
            
            # Extract hull points
            self.hull_points = self.points[hull.vertices]
            # Add closing point to form a closed loop
            self.hull_points = np.append(self.hull_points, [self.hull_points[0]], axis=0)
            
            # Calculate hull area
            hull_area = hull.volume  # In 2D, volume means area
            
            # Update hull info
            self.hull_points_label.setText(f"Hull vertices: {len(hull.vertices)}")
            self.hull_area_label.setText(f"Hull area: {hull_area:.2f}")
            
            # Visualize the hull
            self._plot_hull(self.points, self.hull_points)
            
            self.statusBar.showMessage(f"Computed convex hull with {len(hull.vertices)} vertices")
            
            # Clear any previous results from later steps
            self.segments = None
            self.triangulation_result = None
            
            # Auto-switch to the hull tab
            self.tab_widget.setCurrentIndex(1)
            
        except Exception as e:
            self.statusBar.showMessage(f"Error computing hull: {str(e)}")
            logger.error(f"Error computing hull: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error computing hull: {str(e)}")

    def _plot_hull(self, points, hull_points):
        """Visualize the hull using PyVista or plain Qt
        
        Args:
            points: All input points
            hull_points: Points of the convex hull
        """
        if points is None or len(points) == 0 or hull_points is None or len(hull_points) == 0:
            return
        
        # Clear existing visualization
        for i in reversed(range(self.hull_viz_layout.count())): 
            self.hull_viz_layout.itemAt(i).widget().setParent(None)
        
        # If PyVista is available, use 3D visualization
        if HAVE_PYVISTA:
            try:
                # Create a new plotter for the hull
                frame = QFrame()
                layout = QVBoxLayout(frame)
                
                hull_plotter = QtInteractor(frame)
                layout.addWidget(hull_plotter.interactor)
                
                # Set background color
                hull_plotter.set_background("#383F51")  # Dark blue background
                
                # Convert 2D points to 3D
                points_3d = np.zeros((len(points), 3))
                points_3d[:, 0] = points[:, 0]
                points_3d[:, 1] = points[:, 1]
                
                # Do the same for hull points
                hull_3d = np.zeros((len(hull_points), 3))
                hull_3d[:, 0] = hull_points[:, 0]
                hull_3d[:, 1] = hull_points[:, 1]
                
                # Create point cloud with different colors for hull vertices
                point_cloud = pv.PolyData(points_3d)
                
                # Create different colors for points
                point_colors = np.full(len(points), 'blue')
                
                # Find indices of hull points in the original points array
                for hp in hull_points[:-1]:  # Exclude the closing point
                    for i, p in enumerate(points):
                        if np.array_equal(hp, p):
                            point_colors[i] = 'red'
                            break
                
                # Add points to plotter with colors
                hull_plotter.add_points(point_cloud,
                                       scalars=point_colors,
                                       point_size=5,
                                       render_points_as_spheres=True)
                
                # Create line for hull boundary
                hull_lines = pv.PolyData()
                hull_lines.points = hull_3d
                
                # Create lines connecting consecutive hull points
                line_indices = np.array([[i, i+1] for i in range(len(hull_3d)-1)])
                hull_lines.lines = np.hstack([np.full((len(line_indices), 1), 2), line_indices])
                
                # Add hull line to plotter
                hull_plotter.add_mesh(hull_lines, color='red', line_width=3)
                
                # Add text and axes
                hull_plotter.add_text(f"Convex Hull: {len(hull_points)-1} vertices", 
                                      position='upper_edge', font_size=12, color='white')
                hull_plotter.show_axes()
                hull_plotter.reset_camera()
                
                # Add to layout
                self.hull_viz_layout.addWidget(frame)
                
                # Store reference
                self.hull_plotter = hull_plotter
                
            except Exception as e:
                logger.error(f"Error creating PyVista hull visualization: {e}")
                # Fall back to basic visualization if PyVista fails
                self._create_basic_hull_visualization(points, hull_points)
        else:
            # Create basic hull visualization using Qt
            self._create_basic_hull_visualization(points, hull_points)
    
    def _create_basic_hull_visualization(self, points, hull_points):
        """Create a basic visualization of the hull using Qt
        
        Args:
            points: All input points
            hull_points: Points of the convex hull
        """
        # Create a QLabel with text info
        info_label = QLabel(f"Convex Hull: {len(hull_points)-1} vertices (PyVista not available for 3D view)")
        info_label.setAlignment(Qt.AlignCenter)
        self.hull_viz_layout.addWidget(info_label)
        
        # TODO: Add actual 2D visualization with Qt widgets
        # For now, just display a placeholder
        placeholder = QLabel("2D hull visualization will be implemented later")
        placeholder.setAlignment(Qt.AlignCenter)
        self.hull_viz_layout.addWidget(placeholder)
    
    def compute_segments(self):
        """Compute the segmentation of the hull boundary"""
        self.statusBar.showMessage("Computing segments...")
        
        # Check if we have a hull
        if self.hull_points is None or len(self.hull_points) < 4:  # 3 vertices + 1 closing point
            self.statusBar.showMessage("Error: Compute convex hull first")
            QMessageBox.warning(self, "Error", "Please compute the convex hull first")
            return
        
        # Get the segmentation parameters from the UI
        try:
            segment_length = self.segment_length_spin.value()
            segment_density = self.segment_density_spin.value()
            
            # Check if segment length is valid
            if segment_length <= 0:
                self.segment_length_spin.setValue(1.0)
                segment_length = 1.0
            
            # Extract the hull boundary (excluding the closing point)
            hull_boundary = self.hull_points[:-1]
            
            # Calculate hull perimeter to determine a good segment size
            # If user-provided segment length is too large/small relative to the hull,
            # adjust it to ensure good quality
            hull_perimeter = 0
            for i in range(len(hull_boundary)):
                p1 = hull_boundary[i]
                p2 = hull_boundary[(i + 1) % len(hull_boundary)]
                hull_perimeter += np.linalg.norm(p2 - p1)
            
            # Recommended number of segments for good quality
            recommended_segments = 20 * segment_density
            optimal_segment_length = hull_perimeter / recommended_segments
            
            # Only adjust if user value is very different from optimal
            if segment_length > optimal_segment_length * 3 or segment_length < optimal_segment_length * 0.3:
                logger.info(f"Adjusting segment length from {segment_length} to {optimal_segment_length}")
                segment_length = optimal_segment_length
                self.segment_length_spin.setValue(segment_length)
            
            # Generate segments along the hull boundary with uniform distribution
            segments = []
            segment_lengths = []
            
            for i in range(len(hull_boundary)):
                p1 = hull_boundary[i]
                p2 = hull_boundary[(i + 1) % len(hull_boundary)]
                
                # Calculate distance between points
                dist = np.linalg.norm(p2 - p1)
                
                # Determine how many segments to create - ensure at least 1
                num_segments = max(1, int(np.ceil(dist / segment_length)))
                
                # Create segments with even spacing
                for j in range(num_segments):
                    t1 = j / num_segments
                    t2 = (j + 1) / num_segments
                    segment_start = p1 + t1 * (p2 - p1)
                    segment_end = p1 + t2 * (p2 - p1)
                    
                    segments.append([segment_start, segment_end])
                    segment_lengths.append(np.linalg.norm(segment_end - segment_start))
            
            # Store the segments
            self.segments = np.array(segments)
            
            # Calculate average segment length
            avg_segment_length = np.mean(segment_lengths) if segment_lengths else 0
            
            # Update segment info
            self.num_segments_label.setText(f"Segments: {len(segments)}")
            self.avg_segment_length_label.setText(f"Avg length: {avg_segment_length:.2f}")
            
            # Visualize the segments
            self._plot_segments(self.points, self.hull_points, self.segments)
            
            self.statusBar.showMessage(f"Computed {len(segments)} segments, avg length: {avg_segment_length:.2f}")
            
            # Clear any previous results from later steps
            self.triangulation_result = None
            
            # Auto-switch to the segments tab
            self.tab_widget.setCurrentIndex(2)
            
        except Exception as e:
            self.statusBar.showMessage(f"Error computing segments: {str(e)}")
            logger.error(f"Error computing segments: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error computing segments: {str(e)}")

    def _plot_segments(self, points, hull_points, segments):
        """Visualize the segments using PyVista or plain Qt
        
        Args:
            points: All input points
            hull_points: Points of the convex hull
            segments: Line segments
        """
        if points is None or len(points) == 0 or segments is None or len(segments) == 0:
            return
        
        # Clear existing visualization
        for i in reversed(range(self.segment_viz_layout.count())): 
            self.segment_viz_layout.itemAt(i).widget().setParent(None)
        
        # If PyVista is available, use 3D visualization
        if HAVE_PYVISTA:
            try:
                # Create a new plotter for the segments
                frame = QFrame()
                layout = QVBoxLayout(frame)
                
                segment_plotter = QtInteractor(frame)
                layout.addWidget(segment_plotter.interactor)
                
                # Set background color
                segment_plotter.set_background("#383F51")  # Dark blue background
                
                # Convert 2D points to 3D
                points_3d = np.zeros((len(points), 3))
                points_3d[:, 0] = points[:, 0]
                points_3d[:, 1] = points[:, 1]
                
                # Convert hull points to 3D
                hull_3d = np.zeros((len(hull_points), 3))
                hull_3d[:, 0] = hull_points[:, 0]
                hull_3d[:, 1] = hull_points[:, 1]
                
                # Create point cloud
                point_cloud = pv.PolyData(points_3d)
                
                # Create different colors for points
                point_colors = np.full(len(points), 'blue')
                
                # Find indices of hull points in the original points array
                for hp in hull_points[:-1]:  # Exclude the closing point
                    for i, p in enumerate(points):
                        if np.array_equal(hp, p):
                            point_colors[i] = 'red'
                            break
                
                # Add points to plotter with colors
                segment_plotter.add_points(point_cloud,
                                    scalars=point_colors,
                                    point_size=5,
                                    render_points_as_spheres=True,
                                    opacity=0.5)  # Semi-transparent to see segments better
                
                # Create line for hull boundary
                hull_lines = pv.PolyData()
                hull_lines.points = hull_3d
                hull_indices = np.array([[i, i+1] for i in range(len(hull_3d)-1)])
                hull_lines.lines = np.hstack([np.full((len(hull_indices), 1), 2), hull_indices])
                
                # Add hull line to plotter
                segment_plotter.add_mesh(hull_lines, color='red', line_width=2, opacity=0.5)
                
                # Create segment lines
                all_segment_points = []
                all_lines = []
                point_index = 0
                
                for segment in segments:
                    # Convert segment points to 3D
                    p1_3d = np.array([segment[0][0], segment[0][1], 0])
                    p2_3d = np.array([segment[1][0], segment[1][1], 0])
                    
                    # Add to points and lines
                    all_segment_points.extend([p1_3d, p2_3d])
                    all_lines.append([point_index, point_index + 1])
                    point_index += 2
                
                if all_segment_points:
                    segment_lines = pv.PolyData()
                    segment_lines.points = np.array(all_segment_points)
                    segment_indices = np.array(all_lines)
                    segment_lines.lines = np.hstack([np.full((len(segment_indices), 1), 2), segment_indices])
                    
                    # Add segments to plotter
                    segment_plotter.add_mesh(segment_lines, color='green', line_width=2)
                
                # Add text and axes
                segment_plotter.add_text(f"Segmentation: {len(segments)} segments", 
                                   position='upper_edge', font_size=12, color='white')
                segment_plotter.show_axes()
                segment_plotter.reset_camera()
                
                # Add to layout
                self.segment_viz_layout.addWidget(frame)
                
                # Store reference
                self.segment_plotter = segment_plotter
                
            except Exception as e:
                logger.error(f"Error creating PyVista segment visualization: {e}")
                # Fall back to basic visualization if PyVista fails
                self._create_basic_segment_visualization(points, hull_points, segments)
        else:
            # Create basic segment visualization using Qt
            self._create_basic_segment_visualization(points, hull_points, segments)

    def _create_basic_segment_visualization(self, points, hull_points, segments):
        """Create a basic visualization of the segments using Qt
        
        Args:
            points: All input points
            hull_points: Points of the convex hull
            segments: Line segments
        """
        # Create a QLabel with text info
        info_label = QLabel(f"Segmentation: {len(segments)} segments (PyVista not available for 3D view)")
        info_label.setAlignment(Qt.AlignCenter)
        self.segment_viz_layout.addWidget(info_label)
        
        # TODO: Add actual 2D visualization with Qt widgets
        # For now, just display a placeholder
        placeholder = QLabel("2D segment visualization will be implemented later")
        placeholder.setAlignment(Qt.AlignCenter)
        self.segment_viz_layout.addWidget(placeholder)

    def export_results(self):
        """Export triangulation results to a file"""
        self.statusBar.showMessage("Exporting results...")
        
        if self.triangulation_result is None:
            self.statusBar.showMessage("Error: No triangulation results to export")
            QMessageBox.warning(self, "Export Error", "No triangulation results to export. Please run triangulation first.")
            return
        
        try:
            # Create a dialog to select export options
            dialog = QDialog(self)
            dialog.setWindowTitle("Export Options")
            dialog.setMinimumWidth(300)
            
            layout = QVBoxLayout(dialog)
            
            # Format selection
            format_group = QGroupBox("Export Format")
            format_layout = QVBoxLayout(format_group)
            
            # Format radio buttons
            csv_radio = QCheckBox("CSV Files (*.csv)")
            csv_radio.setChecked(True)
            format_layout.addWidget(csv_radio)
            
            obj_radio = QCheckBox("OBJ Files (*.obj)")
            obj_radio.setChecked(False)
            format_layout.addWidget(obj_radio)
            
            stl_radio = QCheckBox("STL Files (*.stl)")
            stl_radio.setChecked(False)
            format_layout.addWidget(stl_radio)
            
            ply_radio = QCheckBox("PLY Files (*.ply)")
            ply_radio.setChecked(False)
            format_layout.addWidget(ply_radio)
            
            layout.addWidget(format_group)
            
            # 3D export options
            export_3d_group = QGroupBox("3D Export Options")
            export_3d_layout = QFormLayout(export_3d_group)
            
            # Checkbox for 3D export
            export_3d_check = QCheckBox("Export as 3D Surface")
            export_3d_check.setChecked(True)
            export_3d_layout.addRow(export_3d_check)
            
            # Height scale for 3D
            height_scale_spin = QDoubleSpinBox()
            height_scale_spin.setRange(0, 10)
            height_scale_spin.setValue(self.height_factor)
            height_scale_spin.setSingleStep(0.1)
            export_3d_layout.addRow("Height Scale:", height_scale_spin)
            
            # Disable height scale when not exporting 3D
            export_3d_check.toggled.connect(height_scale_spin.setEnabled)
            
            layout.addWidget(export_3d_group)
            
            # Add buttons
            button_box = QHBoxLayout()
            export_btn = QPushButton("Export")
            export_btn.clicked.connect(dialog.accept)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            
            button_box.addWidget(export_btn)
            button_box.addWidget(cancel_btn)
            layout.addLayout(button_box)
            
            # Show dialog
            result = dialog.exec_()
            
            # Check if user clicked Export
            if result == QDialog.Accepted:
                # Create a list of selected formats
                selected_formats = []
                if csv_radio.isChecked():
                    selected_formats.append(("CSV Files", "*.csv"))
                if obj_radio.isChecked():
                    selected_formats.append(("OBJ Files", "*.obj"))
                if stl_radio.isChecked():
                    selected_formats.append(("STL Files", "*.stl"))
                if ply_radio.isChecked():
                    selected_formats.append(("PLY Files", "*.ply"))
                
                # Check if any format was selected
                if not selected_formats:
                    self.statusBar.showMessage("Error: No export format selected")
                    QMessageBox.warning(self, "Export Error", "Please select at least one export format.")
                    return
                    
                # Create filter string for file dialog
                filter_string = ";;".join([f"{desc} ({ext})" for desc, ext in selected_formats])
                
                # Get the output file
                output_file, selected_filter = QFileDialog.getSaveFileName(
                    self, "Export Triangulation", "", filter_string)
                    
                if output_file:
                    # Get selected extension from filter
                    selected_ext = None
                    for desc, ext in selected_formats:
                        if desc in selected_filter:
                            selected_ext = ext[1:]  # Remove the *
                            break
                    
                    # Ensure the filename has the correct extension
                    if not output_file.endswith(selected_ext):
                        output_file += selected_ext
                    
                    # Get data
                    triangles = self.triangulation_result['triangles']
                    vertices = self.triangulation_result['vertices']
                    
                    # Check if we should export as 3D
                    is_3d_export = export_3d_check.isChecked()
                    height_factor = height_scale_spin.value() if is_3d_export else 0
                    
                    # Export based on format
                    if selected_ext == '.csv':
                        self._export_csv(output_file, vertices, triangles)
                    elif selected_ext == '.obj':
                        self._export_obj(output_file, vertices, triangles, height_factor)
                    elif selected_ext == '.stl':
                        self._export_stl(output_file, vertices, triangles, height_factor)
                    elif selected_ext == '.ply':
                        self._export_ply(output_file, vertices, triangles, height_factor)
                    
                    self.statusBar.showMessage(f"Exported triangulation to {output_file}")
                else:
                    self.statusBar.showMessage("Export cancelled")
            else:
                self.statusBar.showMessage("Export cancelled")
            
        except Exception as e:
            self.statusBar.showMessage(f"Error during export: {str(e)}")
            logger.error(f"Error during export: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Error exporting triangulation: {str(e)}")

    def _export_csv(self, filename, vertices, triangles):
        """Export triangulation to CSV format
        
        Args:
            filename: Output filename
            vertices: Vertex array
            triangles: Triangle array
        """
        try:
            # Write vertices to CSV
            vertices_file = filename.replace('.csv', '_vertices.csv')
            np.savetxt(vertices_file, vertices, delimiter=',', header='x,y', comments='')
            
            # Write triangles to CSV (index-based)
            triangles_file = filename.replace('.csv', '_triangles.csv')
            np.savetxt(triangles_file, triangles, delimiter=',', fmt='%d', header='v1,v2,v3', comments='')
            
            logger.info(f"Exported vertices to {vertices_file} and triangles to {triangles_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise

    def _export_obj(self, filename, vertices, triangles, height_factor=0):
        """Export triangulation to OBJ format
        
        Args:
            filename: Output filename
            vertices: Vertex array
            triangles: Triangle array
            height_factor: Height factor for 3D export
        """
        try:
            with open(filename, 'w') as f:
                # Write header
                f.write("# MeshIt triangulation export\n")
                f.write(f"# Vertices: {len(vertices)}, Triangles: {len(triangles)}\n")
                
                # Calculate Z values for 3D export
                if height_factor > 0:
                    # Calculate Z values based on distance from center (creates a dome effect)
                    center = np.mean(vertices, axis=0)
                    max_distance = 0
                    
                    # Find max distance for normalization
                    for vertex in vertices:
                        dx = vertex[0] - center[0]
                        dy = vertex[1] - center[1]
                        distance = np.sqrt(dx**2 + dy**2)
                        max_distance = max(max_distance, distance)
                    
                    # Write vertices with Z values
                    for vertex in vertices:
                        dx = vertex[0] - center[0]
                        dy = vertex[1] - center[1]
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        # Calculate Z value (dome shape)
                        z = 0
                        if max_distance > 0:
                            normalized_distance = distance / max_distance
                            z = height_factor * (1 - normalized_distance**2)
                        
                        f.write(f"v {vertex[0]} {vertex[1]} {z}\n")
                else:
                    # Write vertices (flat - Z=0)
                    for vertex in vertices:
                        f.write(f"v {vertex[0]} {vertex[1]} 0\n")
                
                # Write triangles (OBJ uses 1-indexed vertices)
                for triangle in triangles:
                    f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
                
            logger.info(f"Exported triangulation to OBJ: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to OBJ: {e}")
            raise

    def _export_stl(self, filename, vertices, triangles, height_factor=0):
        """Export triangulation to STL format
        
        Args:
            filename: Output filename
            vertices: Vertex array
            triangles: Triangle array
            height_factor: Height factor for 3D export
        """
        if not HAVE_PYVISTA:
            raise ImportError("PyVista is required for STL export")
        
        try:
            # Convert 2D vertices to 3D
            vertices_3d = np.zeros((len(vertices), 3))
            vertices_3d[:, 0] = vertices[:, 0]
            vertices_3d[:, 1] = vertices[:, 1]
            
            if height_factor > 0:
                # Calculate Z values based on distance from center (creates a dome effect)
                center = np.mean(vertices, axis=0)
                for i, vertex in enumerate(vertices):
                    dx = vertex[0] - center[0]
                    dy = vertex[1] - center[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Calculate Z value (dome shape)
                    max_distance = np.max([np.linalg.norm(v - center) for v in vertices])
                    if max_distance > 0:
                        normalized_distance = distance / max_distance
                        z_value = height_factor * (1 - normalized_distance**2)
                        vertices_3d[i, 2] = z_value
            
            # Create a mesh from the triangulation
            mesh = pv.PolyData(vertices_3d, 
                             np.hstack([np.full((len(triangles), 1), 3), triangles]))
            
            # Save to STL
            mesh.save(filename)
            
            logger.info(f"Exported triangulation to STL: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to STL: {e}")
            raise

    def _export_ply(self, filename, vertices, triangles, height_factor=0):
        """Export triangulation to PLY format
        
        Args:
            filename: Output filename
            vertices: Vertex array
            triangles: Triangle array
            height_factor: Height factor for 3D export
        """
        if not HAVE_PYVISTA:
            raise ImportError("PyVista is required for PLY export")
        
        try:
            # Convert 2D vertices to 3D
            vertices_3d = np.zeros((len(vertices), 3))
            vertices_3d[:, 0] = vertices[:, 0]
            vertices_3d[:, 1] = vertices[:, 1]
            
            if height_factor > 0:
                # Calculate Z values based on distance from center (creates a dome effect)
                center = np.mean(vertices, axis=0)
                for i, vertex in enumerate(vertices):
                    dx = vertex[0] - center[0]
                    dy = vertex[1] - center[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Calculate Z value (dome shape)
                    max_distance = np.max([np.linalg.norm(v - center) for v in vertices])
                    if max_distance > 0:
                        normalized_distance = distance / max_distance
                        z_value = height_factor * (1 - normalized_distance**2)
                        vertices_3d[i, 2] = z_value
            
            # Create a mesh from the triangulation
            mesh = pv.PolyData(vertices_3d, 
                             np.hstack([np.full((len(triangles), 1), 3), triangles]))
            
            # Save to PLY
            mesh.save(filename)
            
            logger.info(f"Exported triangulation to PLY: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to PLY: {e}")
            raise

    def run_triangulation(self):
        """Run triangulation on the segments"""
        self.statusBar.showMessage("Running triangulation...")
        
        # Check if we have segments
        if self.segments is None or len(self.segments) == 0:
            self.statusBar.showMessage("Error: Compute segments first")
            QMessageBox.warning(self, "Error", "Please compute segments first")
            return
        
        try:
            # Get triangulation parameters from UI
            min_angle = self.min_angle_spin.value()
            uniform = self.uniform_check.isChecked()
            use_features = self.use_feature_points_check.isChecked()
            num_features = self.num_features_spin.value() if use_features else 0
            feature_size = self.feature_size_spin.value()
            area_max = 1.0 / self.base_size_factor_spin.value()  # Convert density to area
            
            # Check if the Triangle wrapper is available
            if not HAVE_DIRECT_WRAPPER:
                self.statusBar.showMessage("Error: Triangle wrapper not available")
                QMessageBox.critical(self, "Error", 
                                   "The Triangle wrapper is not available. Cannot perform triangulation.")
                return
            
            # Extract unique points from segments (including interior points)
            all_points = self.points
            
            # Extract segment endpoints as additional constraints
            segment_endpoints = []
            for segment in self.segments:
                segment_endpoints.extend([segment[0], segment[1]])
            
            # Remove duplicates
            unique_segment_points = np.unique(segment_endpoints, axis=0)
            
            # Create the Triangle wrapper
            triangle = DirectTriangleWrapper()
            
            # Prepare constraints (segments)
            constraints = []
            point_map = {}  # Map coordinates to indices
            
            # First add all interior points
            input_points = []
            for i, point in enumerate(all_points):
                point_tuple = tuple(point)
                if point_tuple not in point_map:
                    point_map[point_tuple] = len(input_points)
                    input_points.append(point)
            
            # Then add any segment endpoints that aren't already included
            for point in unique_segment_points:
                point_tuple = tuple(point)
                if point_tuple not in point_map:
                    point_map[point_tuple] = len(input_points)
                    input_points.append(point)
            
            # Create constraints from segments
            for segment in self.segments:
                start_tuple = tuple(segment[0])
                end_tuple = tuple(segment[1])
                
                if start_tuple in point_map and end_tuple in point_map:
                    constraints.append((point_map[start_tuple], point_map[end_tuple]))
            
            # Convert input points to numpy array
            input_points = np.array(input_points)
            
            # Generate feature points if requested
            feature_points = []
            if use_features and num_features > 0:
                # Use convex hull center for features
                hull_center = np.mean(self.hull_points[:-1], axis=0)
                
                # Create features around center
                radius = feature_size * np.min([
                    np.max(input_points[:, 0]) - np.min(input_points[:, 0]),
                    np.max(input_points[:, 1]) - np.min(input_points[:, 1])
                ]) / 5  # Scale based on domain size
                
                for i in range(num_features):
                    angle = 2 * np.pi * i / num_features
                    feature_x = hull_center[0] + radius * np.cos(angle)
                    feature_y = hull_center[1] + radius * np.sin(angle)
                    feature_points.append([feature_x, feature_y])
                
                # Add features to input points and set them as feature points
                feature_points_array = np.array(feature_points)
                feature_sizes = np.ones(len(feature_points)) * self.base_size_factor_spin.value() * 0.1
                triangle.set_feature_points(feature_points_array, feature_sizes)
            
            # Build Triangle options string for set_triangle_options or set directly in triangulate
            options_str = f"q{min_angle}a{area_max}p"  # Quality, area constraint, PSLG
            
            # Add uniform option if checked
            if uniform:
                options_str += "u"
            
            # Log options for debugging
            logger.info(f"Triangle options: {options_str}")
            logger.info(f"Input points: {len(input_points)}, constraints: {len(constraints)}")
            
            # Perform triangulation
            start_time = time.time()
            self.statusBar.showMessage("Triangulating...")
            
            # Run the triangulation and get results directly
            result = triangle.triangulate(input_points, constraints, uniform=uniform)
            
            # Check the contents of the result for debugging
            logger.info(f"Triangulation result keys: {list(result.keys())}")
            
            # The DirectTriangleWrapper implementation appears to be returning 
            # results with 'triangles' but not 'vertices'
            # We need to use the input points as vertices in this case
            triangles = None
            for key in ['triangles', 'elements', 'faces']:
                if key in result:
                    triangles = result[key]
                    logger.info(f"Found triangles with key: {key}")
                    break
            
            # If vertices are not in the result, use the input points
            vertices = None
            for key in ['vertices', 'vertex', 'points']:
                if key in result:
                    vertices = result[key]
                    logger.info(f"Found vertices with key: {key}")
                    break
            
            # If vertices are not found in the result, use the input points
            if vertices is None:
                logger.info("No vertices found in result, using input points")
                vertices = input_points
            
            if triangles is None:
                raise ValueError(f"Cannot find triangles in the result. Keys: {list(result.keys())}")
            
            # Measure elapsed time
            elapsed_time = time.time() - start_time
            
            # Store the results
            self.triangulation_result = {
                'triangles': triangles,
                'vertices': vertices,
                'options': options_str,
                'elapsed_time': elapsed_time,
                'feature_points': feature_points if use_features else []
            }
            
            # Update statistics
            self.num_triangles_label.setText(f"Triangles: {len(triangles)}")
            self.num_vertices_label.setText(f"Vertices: {len(vertices)}")
            
            if hasattr(self, 'triangulation_time_label'):
                self.triangulation_time_label.setText(f"Time: {elapsed_time:.3f} sec")
            
            # Calculate and display additional statistics
            if len(triangles) > 0:
                # Calculate edge lengths
                edge_lengths = []
                for tri in triangles:
                    p1 = vertices[tri[0]]
                    p2 = vertices[tri[1]]
                    p3 = vertices[tri[2]]
                    
                    # Calculate edge lengths for this triangle
                    e1 = np.linalg.norm(p2 - p1)
                    e2 = np.linalg.norm(p3 - p2)
                    e3 = np.linalg.norm(p1 - p3)
                    
                    edge_lengths.extend([e1, e2, e3])
                
                # Calculate statistics
                min_edge = np.min(edge_lengths)
                max_edge = np.max(edge_lengths)
                avg_edge = np.mean(edge_lengths)
                uniformity = min_edge / max_edge if max_edge > 0 else 0
                
                # Update UI
                if hasattr(self, 'min_edge_label'):
                    self.min_edge_label.setText(f"Min edge: {min_edge:.3f}")
                if hasattr(self, 'max_edge_label'):
                    self.max_edge_label.setText(f"Max edge: {max_edge:.3f}")
                
                self.mean_edge_label.setText(f"Mean edge: {avg_edge:.3f}")
                self.uniformity_label.setText(f"Uniformity: {uniformity:.3f}")
            
            # Plot the triangulation
            self._plot_triangulation(vertices, triangles)
            
            # Update status
            self.statusBar.showMessage(f"Triangulation completed in {elapsed_time:.3f} seconds. {len(triangles)} triangles created.")
            
            # Auto-switch to the triangulation tab
            self.tab_widget.setCurrentIndex(3)
            
        except Exception as e:
            self.statusBar.showMessage(f"Error running triangulation: {str(e)}")
            logger.error(f"Error in triangulation: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error running triangulation: {str(e)}")

    def _plot_triangulation(self, vertices, triangles):
        """Plot the triangulation results with enhanced 3D visualization
        
        Args:
            vertices: Vertex array (Nx2 array of coordinates)
            triangles: Triangle array (Nx3 array of vertex indices)
        """
        if vertices is None or len(vertices) == 0 or triangles is None or len(triangles) == 0:
            return
        
        # Clear existing visualization
        for i in reversed(range(self.tri_viz_layout.count())): 
            self.tri_viz_layout.itemAt(i).widget().setParent(None)
        
        # If PyVista is available, use enhanced 3D visualization
        if HAVE_PYVISTA:
            try:
                # Create a new plotter for the triangulation
                frame = QFrame()
                layout = QVBoxLayout(frame)
                
                tri_plotter = QtInteractor(frame)
                layout.addWidget(tri_plotter.interactor)
                
                # Set background color - use a gradient for better depth perception
                tri_plotter.set_background([0.1, 0.1, 0.2], [0.3, 0.3, 0.4])
                
                # Enable anti-aliasing for smoother rendering
                tri_plotter.enable_anti_aliasing()
                
                # Convert 2D vertices to 3D with height values for better visualization
                vertices_3d = np.zeros((len(vertices), 3))
                vertices_3d[:, 0] = vertices[:, 0]
                vertices_3d[:, 1] = vertices[:, 1]
                
                # Add height values based on distance from center with self.height_factor
                center = np.mean(vertices, axis=0)
                max_distance = 0
                
                # Find maximum distance for normalization
                for vertex in vertices:
                    distance = np.linalg.norm(vertex - center)
                    max_distance = max(max_distance, distance)
                
                # Apply height values
                for i, vertex in enumerate(vertices):
                    distance = np.linalg.norm(vertex - center)
                    # Create a dome-like shape with smooth transition
                    normalized_distance = distance / max_distance if max_distance > 0 else 0
                    z_value = self.height_factor * (1 - normalized_distance**2)
                    vertices_3d[i, 2] = z_value
                
                # Create a mesh from the triangulation
                mesh = pv.PolyData(vertices_3d, 
                                 np.hstack([np.full((len(triangles), 1), 3), triangles]))
                
                # Calculate mesh normals for better lighting
                mesh.compute_normals(inplace=True)
                
                # Calculate scalar field for coloring (can be height, distance from center, etc.)
                # Use Z values for coloring by default
                scalars = vertices_3d[:, 2]
                
                # Create a smooth colormap for better visualization
                # First add the mesh in wireframe mode for edges
                tri_plotter.add_mesh(mesh, style='wireframe', color='black', 
                                   line_width=1, opacity=0.5)
                
                # Then add the surface mesh with scalar coloring
                tri_plotter.add_mesh(mesh, scalars=scalars, show_edges=False,
                                   cmap='viridis', smooth_shading=True,
                                   specular=0.5, specular_power=15,
                                   ambient=0.3, diffuse=0.8)
                
                # Add points with depth sorting for better appearance
                points_actor = tri_plotter.add_points(vertices_3d, color='red', 
                                                    point_size=6, render_points_as_spheres=True,
                                                    opacity=0.7)
                # Enable depth sorting for points
                if hasattr(points_actor, 'mapper'):
                    points_actor.mapper.SetResolveCoincidentTopology(True)
                
                # Add scalar bar for the colormap
                tri_plotter.add_scalar_bar("Height", vertical=True, position_x=0.05, 
                                         position_y=0.05, width=0.1, height=0.7)
                
                # Add text with statistics
                stats_text = (
                    f"Triangulation Statistics:\n"
                    f"- {len(triangles)} triangles\n"
                    f"- {len(vertices)} vertices\n"
                    f"- Max height: {self.height_factor:.2f}"
                )
                tri_plotter.add_text(stats_text, position='upper_right', 
                                    font_size=10, color='white',
                                    shadow=True)
                
                # Add interactive features
                tri_plotter.add_axes()
                tri_plotter.add_bounding_box(color='gray', opacity=0.3)
                
                # Add orientation widget
                tri_plotter.add_orientation_widget()
                
                # Enable camera position widget
                tri_plotter.show_camera()
                
                # Set up optimal camera position
                tri_plotter.camera_position = [
                    # Position the camera at an angle above the mesh
                    (center[0], center[1], max_distance * 2),
                    # Focus point at the center of the mesh
                    (center[0], center[1], 0),
                    # Camera up direction
                    (0, 1, 0)
                ]
                
                # Reset camera to fit the mesh
                tri_plotter.reset_camera()
                
                # Add to layout
                self.tri_viz_layout.addWidget(frame)
                
                # Store reference for later updates
                self.tri_plotter = tri_plotter
                
            except Exception as e:
                logger.error(f"Error creating PyVista triangulation visualization: {e}", exc_info=True)
                # Fall back to basic visualization if PyVista fails
                self._create_basic_triangulation_visualization(vertices, triangles)
        else:
            # Create basic triangulation visualization using Qt
            self._create_basic_triangulation_visualization(vertices, triangles)

    def update_3d_visualization(self):
        """Update the 3D visualization with enhanced visuals and lighting"""
        if not HAVE_PYVISTA or not hasattr(self, 'plotter'):
            return
        
        # Clear the plotter
        self.plotter.clear()
        
        # Set background color with gradient
        self.plotter.set_background([0.1, 0.1, 0.2], [0.3, 0.3, 0.4])
        
        # Enable anti-aliasing for smoother rendering
        self.plotter.enable_anti_aliasing()
        
        # Check if we have any datasets
        if not self.datasets:
            self.plotter.add_text("No surfaces added yet. Run triangulation and add to 3D view.", 
                               position='upper_edge', font_size=12, color='white', shadow=True)
            self.plotter.reset_camera()
            return
        
        # Calculate bounding box to position datasets
        all_vertices = []
        for data in self.datasets.values():
            if 'vertices' in data and len(data['vertices']) > 0:
                all_vertices.extend(data['vertices'])
        
        if not all_vertices:
            return
            
        all_vertices = np.array(all_vertices)
        
        # Calculate center and bounds for camera positioning
        center = np.mean(all_vertices, axis=0)
        min_bounds = np.min(all_vertices, axis=0)
        max_bounds = np.max(all_vertices, axis=0)
        bounds_size = np.max(max_bounds - min_bounds)
        
        # Setup lighting for better visualization
        self.plotter.add_light(pv.Light(position=(center[0], center[1], bounds_size*2), 
                                      focal_point=center, color='white', 
                                      intensity=0.6, cone_angle=50))
        
        # Add each dataset to the plotter with enhanced visualization
        for i, (name, data) in enumerate(self.datasets.items()):
            if not data['visible']:
                continue
            
            # Extract the data
            triangles = data['triangles']
            vertices = data['vertices']
            color = data['color']
            height_factor = data.get('height_factor', self.height_factor)
            
            if len(triangles) == 0 or len(vertices) == 0:
                continue
            
            # Calculate offset for displaying multiple datasets
            if len(self.datasets) > 1:
                offset = (i - len(self.datasets) / 2) * bounds_size * 0.7
            else:
                offset = 0
            
            # Convert 2D vertices to 3D (apply height factor)
            vertices_3d = np.zeros((len(vertices), 3))
            vertices_3d[:, 0] = vertices[:, 0] + offset
            vertices_3d[:, 1] = vertices[:, 1]
            
            # Calculate Z values for smooth surface
            local_center = np.mean(vertices, axis=0)
            for j, vertex in enumerate(vertices):
                # Calculate distance from center
                dx = vertex[0] - local_center[0]
                dy = vertex[1] - local_center[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Map distance to height (parabolic function for a dome shape)
                max_distance = np.max([np.linalg.norm(v - local_center) for v in vertices])
                if max_distance > 0:
                    normalized_distance = distance / max_distance
                    z_value = height_factor * (1 - normalized_distance**2)
                    vertices_3d[j, 2] = z_value
            
            # Create a mesh from the triangulation
            mesh = pv.PolyData(vertices_3d, 
                             np.hstack([np.full((len(triangles), 1), 3), triangles]))
            
            # Compute normals for better lighting
            mesh.compute_normals(inplace=True)
            
            # Create scalar field for coloring using height
            scalars = vertices_3d[:, 2]
            
            # Add the mesh with enhanced visualization
            # First wireframe for edges
            self.plotter.add_mesh(mesh, style='wireframe', color='black', 
                               line_width=1, opacity=0.3, label=f"{name} (wireframe)")
            
            # Then surface with lighting
            rgb_color = self._color_name_to_rgb(color)
            
            # Different coloring based on dataset index for distinction
            cmap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
            cmap = cmap_options[i % len(cmap_options)]
            
            # Add main surface mesh with improved lighting
            self.plotter.add_mesh(mesh, scalars=scalars, opacity=0.9,
                               show_edges=False, cmap=cmap,
                               ambient=0.3, diffuse=0.8, 
                               specular=0.5, specular_power=15,
                               smooth_shading=True,
                               label=name)
            
            # Add a small label to identify each surface
            text_pos = [
                local_center[0] + offset,
                local_center[1],
                height_factor + 0.5
            ]
            self.plotter.add_point_labels([text_pos], [name], font_size=12, 
                                       bold=True, shadow=True)
        
        # Add legend
        if len(self.datasets) > 1:
            self.plotter.add_legend(bcolor=[0.1, 0.1, 0.1, 0.7], face='circle', 
                                  size=[0.15, 0.15], position_x=0.8, position_y=0.1)
        
        # Add text and axes
        self.plotter.add_text(f"3D View: {len(self.datasets)} surfaces", 
                            position='upper_edge', font_size=14, color='white',
                            shadow=True)
                            
        # Add orientation widget and axes
        self.plotter.add_axes()
        self.plotter.add_orientation_widget()
        self.plotter.show_camera()
        
        # Set optimal camera position
        self.plotter.camera_position = [
            # Position at an angle
            (center[0], center[1] - bounds_size*2, bounds_size*1.5),
            # Focus at center
            (center[0], center[1], 0),
            # Up direction
            (0, 0, 1)
        ]
        
        # Reset camera to show all datasets
        self.plotter.reset_camera()
        
        self.statusBar.showMessage("Updated 3D visualization with enhanced rendering")

    def _color_name_to_rgb(self, color_name):
        """Convert a color name to RGB values
        
        Args:
            color_name: Name of the color (string)
            
        Returns:
            Tuple of RGB values (0-1 range)
        """
        # Define color mappings (extend as needed)
        color_map = {
            'white': (1.0, 1.0, 1.0),
            'skyblue': (0.53, 0.81, 0.92),
            'lightgreen': (0.56, 0.93, 0.56),
            'pink': (1.0, 0.75, 0.8),
            'lightyellow': (1.0, 1.0, 0.88),
            'lightcoral': (0.94, 0.5, 0.5),
            'lightsteelblue': (0.69, 0.77, 0.87),
            'lightseagreen': (0.13, 0.7, 0.67),
            'lightpink': (1.0, 0.71, 0.76),
            'red': (1.0, 0.0, 0.0),
            'green': (0.0, 1.0, 0.0),
            'blue': (0.0, 0.0, 1.0),
            'cyan': (0.0, 1.0, 1.0),
            'magenta': (1.0, 0.0, 1.0),
            'yellow': (1.0, 1.0, 0.0),
        }
        
        # Return RGB values or default to white
        return color_map.get(color_name.lower(), (1.0, 1.0, 1.0))
        
    def _create_basic_triangulation_visualization(self, vertices, triangles):
        """Create a basic visualization of the triangulation using Qt
        
        Args:
            vertices: Vertex array (Nx2 array of coordinates)
            triangles: Triangle array (Nx3 array of vertex indices)
        """
        # Create a QLabel with text info
        info_label = QLabel(f"Triangulation: {len(triangles)} triangles, {len(vertices)} vertices (PyVista not available for 3D view)")
        info_label.setAlignment(Qt.AlignCenter)
        self.tri_viz_layout.addWidget(info_label)
        
        # TODO: Add actual 2D visualization with Qt widgets
        # For now, just display a placeholder
        placeholder = QLabel("2D triangulation visualization will be implemented later")
        placeholder.setAlignment(Qt.AlignCenter)
        self.tri_viz_layout.addWidget(placeholder)

    def update_height_factor(self):
        """Update the height factor for 3D visualization"""
        if hasattr(self, 'height_factor_spin'):
            self.height_factor = self.height_factor_spin.value()
            # Update visualization if available
            self.update_3d_visualization()
            self.statusBar.showMessage(f"Height factor updated to {self.height_factor}")

    def add_current_to_3d_view(self):
        """Add current triangulation result to 3D view as a new surface"""
        if not HAVE_PYVISTA:
            QMessageBox.information(self, "3D View Disabled",
                                  "PyVista not available. Cannot add to 3D view.")
            return
        
        if self.triangulation_result is None:
            QMessageBox.information(self, "No Triangulation", 
                                  "Please run triangulation first before adding to 3D view.")
            return
        
        # Get a name for the dataset
        dataset_name = f"Surface {len(self.datasets) + 1}"
        
        # Store the result in datasets
        self.datasets[dataset_name] = {
            'triangles': self.triangulation_result['triangles'].copy(),
            'vertices': self.triangulation_result['vertices'].copy(),
            'options': self.triangulation_result['options'].copy(),
            'color': self._get_next_color(),
            'visible': True,
            'height_factor': self.height_factor
        }
        
        # Update surfaces count
        self.surfaces_label.setText(f"Surfaces: {len(self.datasets)}")
        
        # Update 3D visualization
        self.update_3d_visualization()
        
        self.statusBar.showMessage(f"Added triangulation as '{dataset_name}' to 3D view")

    def clear_all_surfaces(self):
        """Clear all surfaces from 3D view"""
        if not self.datasets:
            return
        
        # Confirm with user
        reply = QMessageBox.question(self, "Clear Surfaces", 
                                   f"Are you sure you want to clear all {len(self.datasets)} surfaces?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.datasets.clear()
            self.surfaces_label.setText("Surfaces: 0")
            
            # Update visualization
            self.update_3d_visualization()
            
            self.statusBar.showMessage("Cleared all surfaces from 3D view")

    def _get_next_color(self):
        """Get the next color from the color cycle for a new surface"""
        colors = [
            'white', 'skyblue', 'lightgreen', 'pink', 'lightyellow', 
            'lightcoral', 'lightsteelblue', 'lightseagreen', 'lightpink'
        ]
        
        # Return a color based on the number of existing datasets
        return colors[len(self.datasets) % len(colors)]

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeshItQtGUI()
    window.show()
    sys.exit(app.exec_()) 