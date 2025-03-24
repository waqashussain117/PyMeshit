"""
MeshIt Workflow GUI

This application provides a graphical interface for the complete MeshIt workflow,
including file loading, convex hull computation, segmentation, triangulation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import logging
import os
import sys
import time

# Import PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QTabWidget,
                            QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QGroupBox, QRadioButton, QSlider, QLineEdit,
                            QSplitter, QDialog, QFormLayout, QButtonGroup, QMenu, QAction)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

# Import PyVista for 3D visualization
try:
    import pyvista as pv
    from pyvista import examples
    HAVE_PYVISTA = True
    logging.info("Successfully imported PyVista for 3D visualization")
except ImportError:
    HAVE_PYVISTA = False
    logging.warning("PyVista not available. 3D visualization disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MeshIt-Workflow")

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the DirectTriangleWrapper
try:
    from meshit.triangle_direct import DirectTriangleWrapper
    HAVE_DIRECT_WRAPPER = True
    logger.info("Successfully imported DirectTriangleWrapper")
except ImportError as e:
    logger.error(f"Failed to import DirectTriangleWrapper: {e}")
    HAVE_DIRECT_WRAPPER = False

class MeshItWorkflowGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup main window
        self.setWindowTitle("MeshIt Workflow GUI")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create notebook (tabbed interface)
        self.notebook = QTabWidget()
        self.main_layout.addWidget(self.notebook)
        
        # Create tabs for each workflow step
        self.file_tab = QWidget()
        self.hull_tab = QWidget()
        self.segment_tab = QWidget()
        self.triangulation_tab = QWidget()
        
        self.notebook.addTab(self.file_tab, "1. Load Data")
        self.notebook.addTab(self.hull_tab, "2. Convex Hull")
        self.notebook.addTab(self.segment_tab, "3. Segmentation")
        self.notebook.addTab(self.triangulation_tab, "4. Triangulation")
        
        # Initialize data containers
        self.points = None
        self.hull_points = None
        self.segments = None
        self.triangulation_result = None
        
        # Initialize PyVista 3D visualization
        self.view_3d_enabled = HAVE_PYVISTA
        self.current_plotter = None
        self.height_factor = 0.2
        
        # Set up tabs
        self._setup_file_tab()
        self._setup_hull_tab()
        self._setup_segment_tab()
        self._setup_triangulation_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Create menu bar
        self._create_menu_bar()
    
    def _create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Data", self)
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)
        
        generate_action = QAction("Generate Test Data", self)
        generate_action.triggered.connect(self.generate_test_data)
        file_menu.addAction(generate_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Visualization menu
        viz_menu = menubar.addMenu("Visualization")
        if self.view_3d_enabled:
            show_3d_action = QAction("Show 3D View", self)
            show_3d_action.triggered.connect(self.show_3d_view)
            viz_menu.addAction(show_3d_action)
            
            viz_menu.addSeparator()
            
            # 3D height factor submenu
            height_menu = viz_menu.addMenu("3D Height Factor")
            for height in [0.0, 0.1, 0.2, 0.5, 1.0]:
                height_action = QAction(f"Height Factor: {height}", self)
                height_action.triggered.connect(lambda checked, h=height: self._set_height_factor(h))
                height_menu.addAction(height_action)
        else:
            disabled_action = QAction("3D Visualization Disabled", self)
            disabled_action.setEnabled(False)
            viz_menu.addAction(disabled_action)
            
            install_action = QAction("Install PyVista for 3D features", self)
            install_action.setEnabled(False)
            viz_menu.addAction(install_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_file_tab(self):
        """Sets up the file loading tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.file_tab)
        
        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        
        # -- File Loading Controls --
        file_group = QGroupBox("Data Import")
        file_layout = QVBoxLayout(file_group)
        
        # Buttons for loading files
        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)
        
        # Test data generation
        test_btn = QPushButton("Generate Test Data")
        test_btn.clicked.connect(self.generate_test_data)
        file_layout.addWidget(test_btn)
        
        control_layout.addWidget(file_group)
        
        # -- Statistics --
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Number of points
        self.points_label = QLabel("Points: 0")
        stats_layout.addWidget(self.points_label)
        
        # Bounding box info
        self.bounds_label = QLabel("Bounds: N/A")
        stats_layout.addWidget(self.bounds_label)
        
        control_layout.addWidget(stats_group)
        control_layout.addStretch()
        
        tab_layout.addWidget(control_panel)
        
        # --- Visualization Area (right side) ---
        viz_group = QGroupBox("Point Cloud Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Create container for visualization
        self.file_viz_frame = QWidget()
        self.file_viz_layout = QVBoxLayout(self.file_viz_frame)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Load data to visualize points in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nLoad data to visualize points in 2D."
            
        self.file_viz_placeholder = QLabel(placeholder_text)
        self.file_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.file_viz_layout.addWidget(self.file_viz_placeholder)
        
        viz_layout.addWidget(self.file_viz_frame)
        tab_layout.addWidget(viz_group, 1)  # 1 = stretch factor
    
    def _setup_hull_tab(self):
        """Sets up the convex hull tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.hull_tab)
        
        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        
        # -- Hull Controls --
        hull_group = QGroupBox("Hull Controls")
        hull_layout = QVBoxLayout(hull_group)
        
        # Compute hull button
        compute_btn = QPushButton("Compute Convex Hull")
        compute_btn.clicked.connect(self.compute_hull)
        hull_layout.addWidget(compute_btn)
        
        control_layout.addWidget(hull_group)
        
        # -- Statistics --
        stats_group = QGroupBox("Hull Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Hull points count
        self.hull_points_label = QLabel("Hull vertices: 0")
        stats_layout.addWidget(self.hull_points_label)
        
        # Hull area
        self.hull_area_label = QLabel("Hull area: 0.0")
        stats_layout.addWidget(self.hull_area_label)
        
        control_layout.addWidget(stats_group)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(0))
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(2))
        nav_layout.addWidget(next_btn)
        
        control_layout.addLayout(nav_layout)
        control_layout.addStretch()
        
        tab_layout.addWidget(control_panel)
        
        # --- Visualization Area (right side) ---
        viz_group = QGroupBox("Convex Hull Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Create container for visualization
        self.hull_viz_frame = QWidget()
        self.hull_viz_layout = QVBoxLayout(self.hull_viz_frame)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Compute convex hull to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute convex hull to visualize in 2D."
            
        self.hull_viz_placeholder = QLabel(placeholder_text)
        self.hull_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.hull_viz_layout.addWidget(self.hull_viz_placeholder)
        
        viz_layout.addWidget(self.hull_viz_frame)
        tab_layout.addWidget(viz_group, 1)  # 1 = stretch factor
    
    def _setup_segment_tab(self):
        """Sets up the segmentation tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.segment_tab)
        
        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        
        # -- Segmentation Controls --
        segment_group = QGroupBox("Segmentation Controls")
        segment_layout = QVBoxLayout(segment_group)
        
        # Segment length control
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Segment Length:"))
        self.segment_length_input = QLineEdit("1.0")
        length_layout.addWidget(self.segment_length_input)
        segment_layout.addLayout(length_layout)
        
        # Segment density control
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Density:"))
        self.segment_density_slider = QSlider(Qt.Horizontal)
        self.segment_density_slider.setMinimum(50)
        self.segment_density_slider.setMaximum(200)
        self.segment_density_slider.setValue(100)
        density_layout.addWidget(self.segment_density_slider)
        segment_layout.addLayout(density_layout)
        
        # Compute segments button
        compute_btn = QPushButton("Compute Segmentation")
        compute_btn.clicked.connect(self.compute_segments)
        segment_layout.addWidget(compute_btn)
        
        control_layout.addWidget(segment_group)
        
        # -- Statistics --
        stats_group = QGroupBox("Segment Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Number of segments
        self.num_segments_label = QLabel("Segments: 0")
        stats_layout.addWidget(self.num_segments_label)
        
        # Average segment length
        self.avg_segment_length_label = QLabel("Avg length: 0.0")
        stats_layout.addWidget(self.avg_segment_length_label)
        
        control_layout.addWidget(stats_group)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(1))
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(3))
        nav_layout.addWidget(next_btn)
        
        control_layout.addLayout(nav_layout)
        control_layout.addStretch()
        
        tab_layout.addWidget(control_panel)
        
        # --- Visualization Area (right side) ---
        viz_group = QGroupBox("Segmentation Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Create container for visualization
        self.segment_viz_frame = QWidget()
        self.segment_viz_layout = QVBoxLayout(self.segment_viz_frame)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Compute segmentation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute segmentation to visualize in 2D."
            
        self.segment_viz_placeholder = QLabel(placeholder_text)
        self.segment_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.segment_viz_layout.addWidget(self.segment_viz_placeholder)
        
        viz_layout.addWidget(self.segment_viz_frame)
        tab_layout.addWidget(viz_group, 1)  # 1 = stretch factor
    
    def _setup_triangulation_tab(self):
        """Sets up the triangulation tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.triangulation_tab)
        
        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        
        # -- Triangulation Controls --
        tri_group = QGroupBox("Triangulation Controls")
        tri_layout = QVBoxLayout(tri_group)
        
        # Mesh density controls
        density_layout = QFormLayout()
        self.base_size_factor_input = QDoubleSpinBox()
        self.base_size_factor_input.setRange(5.0, 30.0)
        self.base_size_factor_input.setValue(15.0)
        self.base_size_factor_input.setSingleStep(1.0)
        density_layout.addRow("Mesh Density:", self.base_size_factor_input)
        tri_layout.addLayout(density_layout)
        
        # Mesh quality controls
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QFormLayout(quality_group)
        
        # Gradient
        self.gradient_input = QDoubleSpinBox()
        self.gradient_input.setRange(1.0, 3.0)
        self.gradient_input.setValue(1.0)
        self.gradient_input.setSingleStep(0.1)
        quality_layout.addRow("Gradient:", self.gradient_input)
        
        # Min angle
        self.min_angle_input = QDoubleSpinBox()
        self.min_angle_input.setRange(10.0, 30.0)
        self.min_angle_input.setValue(25.0)
        self.min_angle_input.setSingleStep(1.0)
        quality_layout.addRow("Min Angle:", self.min_angle_input)
        
        # Uniform triangulation
        self.uniform_checkbox = QCheckBox()
        self.uniform_checkbox.setChecked(True)
        quality_layout.addRow("Uniform:", self.uniform_checkbox)
        
        tri_layout.addWidget(quality_group)
        
        # Feature points controls
        feature_group = QGroupBox("Feature Points")
        feature_layout = QFormLayout(feature_group)
        
        # Use features
        self.use_feature_points_checkbox = QCheckBox()
        feature_layout.addRow("Use Features:", self.use_feature_points_checkbox)
        
        # Number of features
        self.num_features_input = QSpinBox()
        self.num_features_input.setRange(1, 10)
        self.num_features_input.setValue(3)
        feature_layout.addRow("Count:", self.num_features_input)
        
        # Feature size
        self.feature_size_input = QDoubleSpinBox()
        self.feature_size_input.setRange(0.1, 3.0)
        self.feature_size_input.setValue(1.0)
        self.feature_size_input.setSingleStep(0.1)
        feature_layout.addRow("Size:", self.feature_size_input)
        
        tri_layout.addWidget(feature_group)
        
        # 3D visualization settings
        viz3d_group = QGroupBox("3D Settings")
        viz3d_layout = QFormLayout(viz3d_group)
        
        # Height scale
        self.height_factor_slider = QSlider(Qt.Horizontal)
        self.height_factor_slider.setMinimum(0)
        self.height_factor_slider.setMaximum(100)
        self.height_factor_slider.setValue(20)  # 20% default
        viz3d_layout.addRow("Height Scale:", self.height_factor_slider)
        
        tri_layout.addWidget(viz3d_group)
        
        # Run triangulation button
        run_btn = QPushButton("Run Triangulation")
        run_btn.clicked.connect(self.run_triangulation)
        tri_layout.addWidget(run_btn)
        
        control_layout.addWidget(tri_group)
        
        # -- Statistics --
        stats_group = QGroupBox("Triangulation Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Number of triangles
        self.num_triangles_label = QLabel("Triangles: 0")
        stats_layout.addWidget(self.num_triangles_label)
        
        # Number of vertices
        self.num_vertices_label = QLabel("Vertices: 0")
        stats_layout.addWidget(self.num_vertices_label)
        
        # Mean edge length
        self.mean_edge_label = QLabel("Mean edge: 0.0")
        stats_layout.addWidget(self.mean_edge_label)
        
        # Edge uniformity
        self.uniformity_label = QLabel("Uniformity: 0.0")
        stats_layout.addWidget(self.uniformity_label)
        
        control_layout.addWidget(stats_group)
        
        # Export button
        export_btn = QPushButton("Export Results...")
        export_btn.clicked.connect(self.export_results)
        control_layout.addWidget(export_btn)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(lambda: self.notebook.setCurrentIndex(2))
        nav_layout.addWidget(prev_btn)
        
        control_layout.addLayout(nav_layout)
        control_layout.addStretch()
        
        tab_layout.addWidget(control_panel)
        
        # --- Visualization Area (right side) ---
        viz_group = QGroupBox("Triangulation Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Create container for visualization
        self.tri_viz_frame = QWidget()
        self.tri_viz_layout = QVBoxLayout(self.tri_viz_frame)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Run triangulation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nRun triangulation to visualize in 2D."
            
        self.tri_viz_placeholder = QLabel(placeholder_text)
        self.tri_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.tri_viz_layout.addWidget(self.tri_viz_placeholder)
        
        viz_layout.addWidget(self.tri_viz_frame)
        tab_layout.addWidget(viz_group, 1)  # 1 = stretch factor

    # Event handlers - placeholder implementations
    def load_file(self):
        """Load data from a file"""
        self.statusBar().showMessage("Loading file...")
        
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a point data file",
            "",
            "Text files (*.txt);;Data files (*.dat);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_path:
            self.statusBar().showMessage("File loading canceled")
            return
            
        # Update status with file path
        self.statusBar().showMessage(f"Loading file: {os.path.basename(file_path)}...")
        
        try:
            # Try to read the file
            points = self._read_point_file(file_path)
            
            if points is not None and len(points) > 0:
                self.points = points
                
                # Visualize the loaded points
                self._plot_points(points)
                
                self.statusBar().showMessage(f"Successfully loaded {len(points)} points")
                
                # Clear any previous results from later steps
                self.hull_points = None
                self.segments = None
                self.triangulation_result = None
                
                # Update other views
                self._clear_hull_plot()
                self._clear_segment_plot()
                self._clear_triangulation_plot()
            else:
                self.statusBar().showMessage("Error: No valid points found in file")
                QMessageBox.critical(self, "Error", "No valid points found in file")
        
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {str(e)}")
            logger.error(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def _read_point_file(self, file_path):
        """Read points from a file"""
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
            QMessageBox.critical(self, "Error", f"Error reading file: {str(e)}")
            return None

    def generate_test_data(self):
        """Generate test data"""
        self.statusBar().showMessage("Generating test data...")
        
        # Create a dialog to select the type of test data
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Test Data")
        dialog.setFixedSize(300, 250)
        dialog_layout = QVBoxLayout(dialog)
        
        # Data type selection
        dialog_layout.addWidget(QLabel("Select data type:"))
        
        data_type_group = QButtonGroup(dialog)
        data_types = ["Random Points", "Circle", "Square", "Complex Shape"]
        data_type_buttons = {}
        
        for i, data_type in enumerate(data_types):
            radio = QRadioButton(data_type)
            dialog_layout.addWidget(radio)
            data_type_group.addButton(radio, i)
            data_type_buttons[i] = radio
            if i == 0:  # Select the first option by default
                radio.setChecked(True)
        
        dialog_layout.addWidget(QLabel("Number of points:"))
        
        num_points_input = QSpinBox()
        num_points_input.setRange(10, 1000)
        num_points_input.setValue(100)
        dialog_layout.addWidget(num_points_input)
        
        button_layout = QHBoxLayout()
        generate_btn = QPushButton("Generate")
        cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(generate_btn)
        
        dialog_layout.addLayout(button_layout)
        
        # Connect signals
        cancel_btn.clicked.connect(dialog.reject)
        generate_btn.clicked.connect(dialog.accept)
        
        # Show dialog and get result
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            # Get the selected data type
            selected_id = data_type_group.checkedId()
            if selected_id >= 0:
                data_type = data_types[selected_id]
                num_points = num_points_input.value()
                
                # Generate the selected data type
                if data_type == "Random Points":
                    points = self._generate_random_points(num_points)
                elif data_type == "Circle":
                    points = self._generate_circle_points(num_points)
                elif data_type == "Square":
                    points = self._generate_square_points(num_points)
                elif data_type == "Complex Shape":
                    points = self._generate_complex_shape(num_points)
                else:
                    points = None
                
                if points is not None:
                    self.points = points
                    
                    # Visualize the generated points
                    self._plot_points(points)
                    
                    self.statusBar().showMessage(f"Generated {len(points)} {data_type} points")
                    
                    # Clear any previous results from later steps
                    self.hull_points = None
                    self.segments = None
                    self.triangulation_result = None
                    
                    # Update other views
                    self._clear_hull_plot()
                    self._clear_segment_plot()
                    self._clear_triangulation_plot()
        else:
            self.statusBar().showMessage("Test data generation canceled")
    
    def _generate_random_points(self, num_points=100):
        """Generate random points"""
        # Generate random points in a square
        points = np.random.rand(num_points, 2) * 20 - 10  # Range: -10 to 10
        return points
    
    def _generate_circle_points(self, num_points=100):
        """Generate points in a circle"""
        # Generate random points in a circle
        r = 10.0 * np.sqrt(np.random.rand(num_points))  # Radius: 0 to 10
        theta = np.random.rand(num_points) * 2 * np.pi  # Angle: 0 to 2pi
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.column_stack((x, y))
    
    def _generate_square_points(self, num_points=100):
        """Generate points in a square"""
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
        """Generate complex shape with internal features"""
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

    # Visualization methods
    def _clear_hull_plot(self):
        """Clear the hull plot"""
        # Clear existing visualization
        while self.hull_viz_layout.count():
            item = self.hull_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Compute convex hull to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute convex hull to visualize in 2D."
            
        self.hull_viz_placeholder = QLabel(placeholder_text)
        self.hull_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.hull_viz_layout.addWidget(self.hull_viz_placeholder)
        
        # Reset hull info
        self.hull_points_label.setText("Hull vertices: 0")
        self.hull_area_label.setText("Hull area: 0.0")
    
    def _clear_segment_plot(self):
        """Clear the segmentation plot"""
        # Clear existing visualization
        while self.segment_viz_layout.count():
            item = self.segment_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Compute segmentation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute segmentation to visualize in 2D."
            
        self.segment_viz_placeholder = QLabel(placeholder_text)
        self.segment_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.segment_viz_layout.addWidget(self.segment_viz_placeholder)
        
        # Reset segment info
        self.num_segments_label.setText("Segments: 0")
        self.avg_segment_length_label.setText("Avg length: 0.0")
    
    def _clear_triangulation_plot(self):
        """Clear the triangulation plot"""
        # Clear existing visualization
        while self.tri_viz_layout.count():
            item = self.tri_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Run triangulation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nRun triangulation to visualize in 2D."
            
        self.tri_viz_placeholder = QLabel(placeholder_text)
        self.tri_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.tri_viz_layout.addWidget(self.tri_viz_placeholder)
        
        # Reset triangulation info
        self.num_triangles_label.setText("Triangles: 0")
        self.num_vertices_label.setText("Vertices: 0")
        self.mean_edge_label.setText("Mean edge: 0.0")
        self.uniformity_label.setText("Uniformity: 0.0")
    
    def _plot_points(self, points):
        """Visualize the loaded points using 3D visualization instead of matplotlib
        
        Args:
            points: 2D points to visualize
        """
        if points is None or len(points) == 0:
            return
            
        # Update statistics
        self.points_label.setText(f"Points: {len(points)}")
        
        # Calculate bounds
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        self.bounds_label.setText(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}]")
        
        # Clear existing visualization
        while self.file_viz_layout.count():
            item = self.file_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Create 3D visualization
        if self.view_3d_enabled:
            self._create_3d_visualization(
                self.file_viz_frame, 
                points, 
                title=f"Point Cloud: {len(points)} points"
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            ax.scatter(points[:, 0], points[:, 1], s=5, c='blue', alpha=0.7)
            ax.set_aspect('equal')
            ax.set_title(f"Point Cloud: {len(points)} points")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            padding = (max_x - min_x) * 0.05
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.file_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.file_viz_frame)
            self.file_viz_layout.addWidget(toolbar)

    def compute_hull(self):
        """Compute the convex hull of the loaded points"""
        self.statusBar().showMessage("Computing convex hull...")
        
        if self.points is None or len(self.points) < 3:
            self.statusBar().showMessage("Error: Need at least 3 points to compute hull")
            QMessageBox.critical(self, "Error", "Need at least 3 points to compute hull")
            return
        
        try:
            # Compute convex hull using scipy
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.points)
            
            # Extract hull points
            self.hull_points = self.points[hull.vertices]
            self.hull_points = np.append(self.hull_points, [self.hull_points[0]], axis=0)  # Close the hull
            
            # Calculate hull area
            hull_area = hull.volume  # In 2D, volume means area
            
            # Update hull info
            self.hull_points_label.setText(f"Hull vertices: {len(hull.vertices)}")
            self.hull_area_label.setText(f"Hull area: {hull_area:.2f}")
            
            # Visualize the hull
            self._plot_hull(self.points, self.hull_points)
            
            self.statusBar().showMessage(f"Computed convex hull with {len(hull.vertices)} vertices")
            
            # Clear any previous results from later steps
            self.segments = None
            self.triangulation_result = None
            
            # Update other views
            self._clear_segment_plot()
            self._clear_triangulation_plot()
            
            # Auto-switch to the hull tab
            self.notebook.setCurrentIndex(1)  # Select hull tab (index 1)
            
        except Exception as e:
            self.statusBar().showMessage(f"Error computing hull: {str(e)}")
            logger.error(f"Error computing hull: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error computing hull: {str(e)}")
    
    def _plot_hull(self, points, hull_points):
        """Visualize the hull using 3D visualization instead of matplotlib
        
        Args:
            points: All input points
            hull_points: Points of the convex hull
        """
        if points is None or len(points) == 0 or hull_points is None or len(hull_points) == 0:
            return
            
        # Update statistics
        self.hull_points_label.setText(f"Hull vertices: {len(hull_points)-1}")  # -1 for the closing point
        
        # Calculate hull area (approximate)
        area = 0.0
        for i in range(len(hull_points)-1):
            x1, y1 = hull_points[i]
            x2, y2 = hull_points[i+1]
            area += x1*y2 - x2*y1
        area = abs(area) / 2.0
        self.hull_area_label.setText(f"Hull area: {area:.2f}")
        
        # Clear existing visualization
        while self.hull_viz_layout.count():
            item = self.hull_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Create 3D visualization
        if self.view_3d_enabled:
            # Create closed line segments from hull points
            hull_lines = []
            for i in range(len(hull_points)-1):
                start = hull_points[i]
                end = hull_points[i+1]
                hull_lines.append([start, end])
            
            # Different point colors: blue for regular points, red for hull vertices
            point_colors = np.full(len(points), 'blue')
            
            # Find indices of hull points in the original points array
            for hp in hull_points[:-1]:  # Exclude the closing point
                # Find matching points
                for i, p in enumerate(points):
                    if np.array_equal(hp, p):
                        point_colors[i] = 'red'
                        break
            
            self._create_3d_visualization(
                self.hull_viz_frame, 
                points, 
                title=f"Convex Hull: {len(hull_points)-1} vertices",
                point_colors=point_colors,
                lines=hull_lines
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot all points
            ax.scatter(points[:, 0], points[:, 1], s=5, c='blue', alpha=0.7, label='Points')
            
            # Plot hull
            ax.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2, label='Hull')
            ax.scatter(hull_points[:-1, 0], hull_points[:-1, 1], s=30, c='red', label='Hull Vertices')
            
            ax.set_aspect('equal')
            ax.set_title(f"Convex Hull: {len(hull_points)-1} vertices")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Set limits with some margin
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            
            # Add 10% margin
            margin = 0.1 * np.max(max_coords - min_coords)
            
            ax.set_xlim(min_coords[0] - margin, max_coords[0] + margin)
            ax.set_ylim(min_coords[1] - margin, max_coords[1] + margin)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.hull_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.hull_viz_frame)
            self.hull_viz_layout.addWidget(toolbar)
    
    def _create_3d_visualization(self, parent_frame, points, title="3D Visualization", 
                               point_colors=None, lines=None, triangles=None, feature_points=None):
        """Create a visualization of the data in 3D using PyVista
        
        Args:
            parent_frame: Frame to contain the visualization
            points: Points to visualize (Nx2 or Nx3 array)
            title: Title for the visualization
            point_colors: Optional colors for points
            lines: Optional line segments to draw
            triangles: Optional triangles to draw
            feature_points: Optional feature points to highlight
            
        Returns:
            Frame containing the visualization
        """
        if not HAVE_PYVISTA:
            # Create a message if PyVista is not available
            msg_widget = QWidget()
            msg_layout = QVBoxLayout(msg_widget)
            
            msg = QLabel("PyVista not installed.\nPlease install PyVista for 3D visualization.")
            msg.setAlignment(Qt.AlignCenter)
            msg_layout.addWidget(msg)
            
            parent_frame.layout().addWidget(msg_widget)
            return msg_widget
        
        # Close previous plotter if it exists
        if self.current_plotter is not None:
            try:
                self.current_plotter.close()
            except:
                pass
            
        # Create a frame for the visualization
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)
        
        # Add visualization info header
        info_label = QLabel(title)
        info_label.setAlignment(Qt.AlignCenter)
        vis_layout.addWidget(info_label)
        
        # Convert 2D points to 3D with height variation
        if points.shape[1] == 2:
            # Convert 2D points to 3D
            points_3d = np.zeros((len(points), 3))
            points_3d[:, 0] = points[:, 0]
            points_3d[:, 1] = points[:, 1]
            
            # Add Z values with a nice pattern
            x_scale = 0.05
            y_scale = 0.05
            z_scale = 20.0 * self.height_factor
            
            for i in range(len(points)):
                x, y = points[i]
                z = np.sin(x * x_scale) * np.cos(y * y_scale) * z_scale
                points_3d[i, 2] = z
        else:
            points_3d = points.copy()
            
        try:
            # Create PyVista plotter widget
            from pyvistaqt import QtInteractor
            
            # Create the QVTKRenderWindowInteractor
            viz_frame = QFrame()
            viz_layout = QVBoxLayout(viz_frame)
            
            # Create the interactor
            self.current_plotter = QtInteractor(viz_frame)
            viz_layout.addWidget(self.current_plotter)
            
            # Set background color and size
            self.current_plotter.set_background("#383F51")
            
            # Add point cloud
            if point_colors is None:
                point_colors = np.full(len(points_3d), 'blue')
            
            # Create point cloud
            point_cloud = pv.PolyData(points_3d)
            
            # Add points with colors
            if isinstance(point_colors, np.ndarray) and point_colors.shape == (len(points_3d),):
                point_cloud["colors"] = point_colors
                self.current_plotter.add_mesh(point_cloud, render_points_as_spheres=True, 
                                         point_size=10, scalar_bar_args={'title': 'Colors'})
            else:
                self.current_plotter.add_mesh(point_cloud, color='blue', render_points_as_spheres=True, 
                                         point_size=10)
            
            # Add lines if provided
            if lines is not None and len(lines) > 0:
                try:
                    for i, line in enumerate(lines):
                        # Create 3D line points
                        line_points = np.zeros((2, 3))
                        line_points[0, 0] = line[0][0]
                        line_points[0, 1] = line[0][1]
                        line_points[1, 0] = line[1][0]
                        line_points[1, 1] = line[1][1]
                        
                        # Set Z values
                        x1, y1 = line[0]
                        x2, y2 = line[1]
                        line_points[0, 2] = np.sin(x1 * x_scale) * np.cos(y1 * y_scale) * z_scale
                        line_points[1, 2] = np.sin(x2 * x_scale) * np.cos(y2 * y_scale) * z_scale
                        
                        # Create line
                        line_obj = pv.Line(line_points[0], line_points[1])
                        self.current_plotter.add_mesh(line_obj, color='green', line_width=3)
                except Exception as e:
                    print(f"Error adding lines: {str(e)}")
            
            # Add triangles if provided
            if triangles is not None and len(triangles) > 0:
                try:
                    # Create cells array
                    cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
                    surface = pv.PolyData(points_3d, cells)
                    self.current_plotter.add_mesh(surface, color='#70D6FF', show_edges=True, 
                                              line_width=1, edge_color='#00CCFF', 
                                              specular=0.5, smooth_shading=True)
                except Exception as e:
                    print(f"Error adding triangles: {str(e)}")
            
            # Add feature points if provided
            if feature_points is not None and len(feature_points) > 0:
                try:
                    # Convert feature points to 3D
                    fp_3d = np.zeros((len(feature_points), 3))
                    fp_3d[:, 0] = feature_points[:, 0]
                    fp_3d[:, 1] = feature_points[:, 1]
                    
                    # Add Z values
                    for i in range(len(feature_points)):
                        x, y = feature_points[i]
                        z = np.sin(x * x_scale) * np.cos(y * y_scale) * z_scale
                        fp_3d[i, 2] = z
                    
                    # Create point cloud for feature points
                    fp_cloud = pv.PolyData(fp_3d)
                    self.current_plotter.add_mesh(fp_cloud, color='red', render_points_as_spheres=True, 
                                              point_size=15)
                except Exception as e:
                    print(f"Error adding feature points: {str(e)}")
            
            # Add axes
            self.current_plotter.add_axes()
            
            # Reset camera to show all objects
            self.current_plotter.reset_camera()
            
            # Add controls for adjustment
            controls_layout = QHBoxLayout()
            
            # Height adjustment slider
            controls_layout.addWidget(QLabel("Height:"))
            height_slider = QSlider(Qt.Horizontal)
            height_slider.setMinimum(0)
            height_slider.setMaximum(100)
            height_slider.setValue(int(self.height_factor * 100))
            height_slider.valueChanged.connect(lambda v: self._update_height_in_plotter(v/100.0))
            controls_layout.addWidget(height_slider)
            
            # Add zoom controls
            controls_layout.addWidget(QLabel("Zoom:"))
            zoom_in_btn = QPushButton("+")
            zoom_in_btn.setMaximumWidth(30)
            zoom_in_btn.clicked.connect(lambda: self.current_plotter.camera.zoom(1.5))
            controls_layout.addWidget(zoom_in_btn)
            
            zoom_out_btn = QPushButton("-")
            zoom_out_btn.setMaximumWidth(30)
            zoom_out_btn.clicked.connect(lambda: self.current_plotter.camera.zoom(0.75))
            controls_layout.addWidget(zoom_out_btn)
            
            # Reset view button
            reset_btn = QPushButton("Reset View")
            reset_btn.clicked.connect(lambda: self.current_plotter.reset_camera())
            controls_layout.addWidget(reset_btn)
            
            vis_layout.addLayout(controls_layout)
            vis_layout.addWidget(viz_frame)
            
        except Exception as e:
            # Fallback if QtInteractor fails
            error_msg = QLabel(f"Error creating 3D view: {str(e)}\nTry using the 'Show 3D View' option in the Visualization menu.")
            error_msg.setAlignment(Qt.AlignCenter)
            vis_layout.addWidget(error_msg)
            logger.error(f"Error creating embedded 3D view: {str(e)}")
        
        parent_frame.layout().addWidget(vis_widget)
        return vis_widget
        
    def _update_height_in_plotter(self, height_factor):
        """Update the height factor in the current plotter"""
        self.height_factor = height_factor
        
        # If we have a current plotter, update it
        if hasattr(self, 'current_plotter') and self.current_plotter is not None:
            try:
                # Get the current visualization data
                if self.triangulation_result is not None:
                    vertices = self.triangulation_result['vertices']
                    triangles = self.triangulation_result['triangles']
                    self._plot_triangulation(vertices, triangles, self.hull_points)
                elif self.segments is not None:
                    self._plot_segments(self.points, self.hull_points, self.segments)
                elif self.hull_points is not None:
                    self._plot_hull(self.points, self.hull_points)
                elif self.points is not None:
                    self._plot_points(self.points)
            except Exception as e:
                logger.error(f"Error updating 3D height: {str(e)}")
    
    def _show_interactive_3d(self, points, point_colors=None, lines=None, triangles=None,
                           feature_points=None, title="MeshIt 3D View"):
        """Open an interactive 3D visualization window"""
        if not self.view_3d_enabled:
            QMessageBox.information(self, "PyVista Not Available", 
                                 "PyVista is not available. Please install PyVista for 3D visualization.")
            return
            
        if points is None or len(points) == 0:
            QMessageBox.information(self, "No Data", 
                                 "No data available to visualize.")
            return
        
        # Close previous plotter if it exists
        if hasattr(self, 'pv_plotter') and self.pv_plotter is not None:
            try:
                self.pv_plotter.close()
            except:
                pass
        
        # Create a new plotter
        self.pv_plotter = pv.Plotter(window_size=[800, 600], title=title)
        self.pv_plotter.set_background("#383F51")  # Dark blue background
        
        # Ensure points are 3D
        if points.shape[1] == 2:
            # Convert 2D points to 3D
            points_3d = np.zeros((len(points), 3))
            points_3d[:, 0] = points[:, 0]
            points_3d[:, 1] = points[:, 1]
            
            # Add Z values with a nice pattern
            x_scale = 0.05
            y_scale = 0.05
            z_scale = 20.0 * self.height_factor
            
            for i in range(len(points)):
                x, y = points[i]
                z = np.sin(x * x_scale) * np.cos(y * y_scale) * z_scale
                points_3d[i, 2] = z
        else:
            points_3d = points.copy()
        
        # Create a point cloud
        if point_colors is None:
            point_colors = np.full(len(points_3d), 'blue')
        
        # Create a PolyData with points
        point_cloud = pv.PolyData(points_3d)
        if isinstance(point_colors, np.ndarray) and point_colors.shape == (len(points_3d),):
            point_cloud["colors"] = point_colors
            self.pv_plotter.add_mesh(point_cloud, render_points_as_spheres=True, 
                                  point_size=10, scalar_bar_args={'title': 'Colors'})
        else:
            self.pv_plotter.add_mesh(point_cloud, color='blue', render_points_as_spheres=True, 
                                  point_size=10)
        
        # Add lines if provided
        if lines is not None and len(lines) > 0:
            try:
                for i, line in enumerate(lines):
                    # Create 3D line points
                    line_points = np.zeros((2, 3))
                    line_points[0, 0] = line[0][0]
                    line_points[0, 1] = line[0][1]
                    line_points[1, 0] = line[1][0]
                    line_points[1, 1] = line[1][1]
                    
                    # Set Z values
                    x1, y1 = line[0]
                    x2, y2 = line[1]
                    line_points[0, 2] = np.sin(x1 * x_scale) * np.cos(y1 * y_scale) * z_scale
                    line_points[1, 2] = np.sin(x2 * x_scale) * np.cos(y2 * y_scale) * z_scale
                    
                    # Create line
                    line_obj = pv.Line(line_points[0], line_points[1])
                    self.pv_plotter.add_mesh(line_obj, color='green', line_width=3)
            except Exception as e:
                print(f"Error adding lines: {str(e)}")
        
        # Add triangles if provided
        if triangles is not None and len(triangles) > 0:
            try:
                # Create cells array
                cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
                surface = pv.PolyData(points_3d, cells)
                self.pv_plotter.add_mesh(surface, color='#70D6FF', show_edges=True, 
                                      line_width=1, edge_color='#00CCFF', 
                                      specular=0.5, smooth_shading=True)
            except Exception as e:
                print(f"Error adding triangles: {str(e)}")
        
        # Add feature points if provided
        if feature_points is not None and len(feature_points) > 0:
            try:
                # Convert feature points to 3D
                fp_3d = np.zeros((len(feature_points), 3))
                fp_3d[:, 0] = feature_points[:, 0]
                fp_3d[:, 1] = feature_points[:, 1]
                
                # Add Z values
                for i in range(len(feature_points)):
                    x, y = feature_points[i]
                    z = np.sin(x * x_scale) * np.cos(y * y_scale) * z_scale
                    fp_3d[i, 2] = z
                
                # Create point cloud for feature points
                fp_cloud = pv.PolyData(fp_3d)
                self.pv_plotter.add_mesh(fp_cloud, color='red', render_points_as_spheres=True, 
                                      point_size=15)
            except Exception as e:
                print(f"Error adding feature points: {str(e)}")
        
        # Add axes for reference
        self.pv_plotter.add_axes()
        
        # Add title with statistics
        if title:
            self.pv_plotter.add_text(title, font_size=12, position='upper_edge')
        
        # Show the plotter in a separate window
        self.pv_plotter.show()
    
    def show_3d_view(self):
        """Show 3D view of the triangulation in a separate window"""
        if not self.view_3d_enabled:
            QMessageBox.information(self, "PyVista Not Available", 
                                 "PyVista is not available. Please install PyVista for 3D visualization.")
            return
            
        if not hasattr(self, 'triangulation_result') or self.triangulation_result is None:
            QMessageBox.information(self, "No Triangulation", 
                                 "Please run triangulation first before showing 3D view.")
            return
            
        # Get triangulation data
        vertices = self.triangulation_result['vertices']
        triangles = self.triangulation_result['triangles']
        hull_points = self.hull_points if hasattr(self, 'hull_points') else None
        
        # Create hull lines if available
        hull_lines = None
        if hull_points is not None and len(hull_points) > 2:
            hull_lines = []
            for i in range(len(hull_points)-1):
                start = hull_points[i]
                end = hull_points[i+1]
                hull_lines.append([start, end])
        
        # Show in interactive window
        title = f"MeshIt 3D View - {len(triangles)} triangles"
        self._show_interactive_3d(
            points=vertices, 
            lines=hull_lines,
            triangles=triangles,
            title=title
        )
        
    def _set_height_factor(self, height):
        """Set the height factor for 3D visualization"""
        self.height_factor = height
        self.height_factor_slider.setValue(int(height * 100))
        
        # Update existing visualization if any
        if hasattr(self, 'triangulation_result') and self.triangulation_result is not None:
            vertices = self.triangulation_result['vertices']
            triangles = self.triangulation_result['triangles']
            self._plot_triangulation(vertices, triangles, self.hull_points)
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """
MeshIt Workflow GUI

A graphical interface for the complete MeshIt workflow,
including file loading, convex hull computation, 
segmentation, triangulation, and visualization.

3D Visualization: {status}
"""
        
        status = "Enabled" if self.view_3d_enabled else "Disabled (Install PyVista)"
        about_text = about_text.format(status=status)
        
        QMessageBox.information(self, "About MeshIt Workflow GUI", about_text)

    def compute_segments(self):
        """Compute the segmentation of the convex hull"""
        self.statusBar().showMessage("Computing segments...")
        
        if self.hull_points is None or len(self.hull_points) < 4:  # 3 vertices + 1 closing point
            self.statusBar().showMessage("Error: Compute convex hull first")
            QMessageBox.critical(self, "Error", "Please compute the convex hull first")
            return
        
        # Get the segmentation parameters from the UI
        try:
            segment_length = float(self.segment_length_input.text())
            if segment_length <= 0:
                segment_length = 1.0
                self.segment_length_input.setText("1.0")
        except ValueError:
            segment_length = 1.0
            self.segment_length_input.setText("1.0")
        
        try:
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
            recommended_segments = 20
            optimal_segment_length = hull_perimeter / recommended_segments
            
            # Only adjust if user value is very different from optimal
            if segment_length > optimal_segment_length * 3 or segment_length < optimal_segment_length * 0.3:
                logger.info(f"Adjusting segment length from {segment_length} to {optimal_segment_length}")
                segment_length = optimal_segment_length
                self.segment_length_input.setText(f"{segment_length:.2f}")
            
            # Density factor from slider (0.5 to 2.0 range)
            density_factor = self.segment_density_slider.value() / 100.0
            segment_length /= density_factor
            
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
            avg_segment_length = np.mean(segment_lengths)
            
            # Update segment info
            self.num_segments_label.setText(f"Segments: {len(segments)}")
            self.avg_segment_length_label.setText(f"Avg length: {avg_segment_length:.2f}")
            
            # Visualize the segments
            self._plot_segments(self.points, self.hull_points, self.segments)
            
            self.statusBar().showMessage(f"Computed {len(segments)} segments, avg length: {avg_segment_length:.2f}")
            
            # Clear any previous results from later steps
            self.triangulation_result = None
            
            # Update other views
            self._clear_triangulation_plot()
            
            # Auto-switch to the segments tab
            self.notebook.setCurrentIndex(2)  # Select segments tab (index 2)
            
        except Exception as e:
            self.statusBar().showMessage(f"Error computing segments: {str(e)}")
            logger.error(f"Error computing segments: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error computing segments: {str(e)}")
            
    def _plot_segments(self, points, hull_points, segments):
        """Visualize the segments using 3D visualization instead of matplotlib
        
        Args:
            points: All input points
            hull_points: Points of the convex hull
            segments: Line segments
        """
        if points is None or len(points) == 0 or segments is None or len(segments) == 0:
            return
            
        # Update statistics
        self.num_segments_label.setText(f"Segments: {len(segments)}")
        
        # Calculate average segment length
        segment_lengths = [np.linalg.norm(segment[1] - segment[0]) for segment in segments]
        avg_length = np.mean(segment_lengths)
        self.avg_segment_length_label.setText(f"Avg length: {avg_length:.2f}")
        
        # Clear existing visualization
        while self.segment_viz_layout.count():
            item = self.segment_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            
        # Create 3D visualization
        if self.view_3d_enabled:
            # Extract segment endpoints for visualization
            segment_points = np.vstack([segment[0] for segment in segments] + 
                                      [segment[1] for segment in segments])
            
            # Create unique points for visualization to avoid duplicates
            unique_points, indices = np.unique(segment_points, axis=0, return_inverse=True)
            
            # Create different point colors: blue for regular points, red for hull vertices, green for segment points
            point_colors = np.full(len(points), 'blue')
            
            # Find indices of hull points in the original points array
            for hp in hull_points[:-1]:  # Exclude the closing point
                # Find matching points
                for i, p in enumerate(points):
                    if np.array_equal(hp, p):
                        point_colors[i] = 'red'
                        break
            
            self._create_3d_visualization(
                self.segment_viz_frame, 
                points, 
                title=f"Segmentation: {len(segments)} segments",
                point_colors=point_colors,
                lines=segments
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot all points
            ax.scatter(points[:, 0], points[:, 1], s=5, c='blue', alpha=0.4, label='Points')
            
            # Plot hull
            ax.plot(hull_points[:, 0], hull_points[:, 1], 'r-', alpha=0.5, linewidth=1, label='Hull')
            
            # Plot segments
            for segment in segments:
                ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 
                      'g-', linewidth=1.5)
            
            # Plot segment endpoints
            all_endpoints = np.vstack([segment[0] for segment in segments] + 
                                     [segment[1] for segment in segments])
            ax.scatter(all_endpoints[:, 0], all_endpoints[:, 1], s=15, c='green', 
                     label='Segment Points')
            
            ax.set_aspect('equal')
            ax.set_title(f"Segmentation: {len(segments)} segments")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Set limits with some margin
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            
            # Add 10% margin
            margin = 0.1 * np.max(max_coords - min_coords)
            
            ax.set_xlim(min_coords[0] - margin, max_coords[0] + margin)
            ax.set_ylim(min_coords[1] - margin, max_coords[1] + margin)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.segment_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.segment_viz_frame)
            self.segment_viz_layout.addWidget(toolbar)

    def run_triangulation(self):
        """Run triangulation on the segments and points"""
        self.statusBar().showMessage("Running triangulation...")
        
        if self.segments is None or len(self.segments) < 3:
            self.statusBar().showMessage("Error: Compute segments first")
            QMessageBox.critical(self, "Error", "Please compute segments first")
            return
        
        # Get triangulation parameters from UI
        gradient = self.gradient_input.value()
        min_angle = self.min_angle_input.value()
        base_size_factor = self.base_size_factor_input.value()
        uniform = self.uniform_checkbox.isChecked()
        use_feature_points = self.use_feature_points_checkbox.isChecked()
        
        try:
            start_time = time.time()
            
            # Create a boundary from the hull points (excluding the closing point)
            boundary_points = self.hull_points[:-1]
            
            # Calculate the domain size (bounding box diagonal)
            min_coords = np.min(boundary_points, axis=0)
            max_coords = np.max(boundary_points, axis=0)
            diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
            
            # Use a smaller base size for denser mesh (matching MeshIt's approach)
            # The smaller the divisor, the smaller the triangles will be
            base_size = diagonal / (base_size_factor * 2.0)  # Make triangles twice as small
            
            # Create segments connecting adjacent boundary points
            boundary_segments = []
            for i in range(len(boundary_points)):
                boundary_segments.append([i, (i + 1) % len(boundary_points)])
            
            # Create internal grid points for better triangulation
            grid_density = 15  # Increase this for denser mesh
            x_min, y_min = min_coords
            x_max, y_max = max_coords
            
            grid_spacing = min(x_max - x_min, y_max - y_min) / grid_density
            
            x_points = np.linspace(x_min + grid_spacing, x_max - grid_spacing, grid_density)
            y_points = np.linspace(y_min + grid_spacing, y_max - grid_spacing, grid_density)
            
            # Create grid points
            grid_points = []
            for x in x_points:
                for y in y_points:
                    grid_points.append([x, y])
            
            # Convert to numpy array if any points created
            if grid_points:
                grid_points = np.array(grid_points)
                
                # Add some jitter to avoid too regular grid
                jitter = grid_spacing * 0.2
                grid_points += np.random.uniform(-jitter, jitter, grid_points.shape)
                
                # Filter out points outside the hull
                from matplotlib.path import Path
                hull_path = Path(boundary_points)
                inside_mask = hull_path.contains_points(grid_points)
                grid_points = grid_points[inside_mask]
                
                # Combine boundary and grid points
                all_points = np.vstack([boundary_points, grid_points])
            else:
                all_points = boundary_points
            
            # Create a DirectTriangleWrapper
            from meshit.triangle_direct import DirectTriangleWrapper
            
            triangulator = DirectTriangleWrapper(
                gradient=gradient,
                min_angle=min_angle,
                base_size=base_size
            )
            
            # Add more options for the Triangle library to match MeshIt style
            triangle_options = f"pzq{min_angle}a{base_size*base_size*0.5}"
            triangulator.set_triangle_options(triangle_options)
            
            # Run triangulation
            self.statusBar().showMessage(f"Running triangulation with base size {base_size:.2f}...")
            
            triangulation_result = triangulator.triangulate(
                points=all_points,
                segments=np.array(boundary_segments),
                uniform=True  # Force uniform for MeshIt-like results
            )
            
            # Extract results
            vertices = triangulation_result['vertices']
            triangles = triangulation_result['triangles']
            
            # Store triangulation results
            self.triangulation_result = {
                'vertices': vertices,
                'triangles': triangles,
                'uniform': uniform,
                'gradient': gradient,
                'min_angle': min_angle,
                'base_size': base_size,
                'grid_points': grid_points if grid_points is not None else np.empty((0, 2))
            }
            
            # Update visualization
            self._plot_triangulation(vertices, triangles, self.hull_points)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            self.statusBar().showMessage(f"Completed triangulation: {len(triangles)} triangles, {len(vertices)} vertices in {elapsed_time:.2f}s")
            
            # Auto-switch to the triangulation tab
            self.notebook.setCurrentIndex(3)  # Select triangulation tab (index 3)
            
        except Exception as e:
            self.statusBar().showMessage(f"Error in triangulation: {str(e)}")
            logger.error(f"Error during triangulation: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error in triangulation: {str(e)}")
    
    def _plot_triangulation(self, vertices, triangles, hull_points=None):
        """Plot the triangulation using 3D visualization
        
        Args:
            vertices: Mesh vertices
            triangles: Triangle indices
            hull_points: Optional hull boundary points
        """
        if vertices is None or len(vertices) == 0 or triangles is None or len(triangles) == 0:
            return
            
        # Update statistics
        self.num_triangles_label.setText(f"Triangles: {len(triangles)}")
        self.num_vertices_label.setText(f"Vertices: {len(vertices)}")
        
        # Calculate edge lengths and statistics
        edge_lengths = []
        for tri in triangles:
            v1, v2, v3 = tri
            p1, p2, p3 = vertices[v1], vertices[v2], vertices[v3]
            
            edge_lengths.extend([
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p3 - p2),
                np.linalg.norm(p1 - p3)
            ])
        
        mean_edge = np.mean(edge_lengths)
        edge_std = np.std(edge_lengths)
        uniformity = edge_std / mean_edge if mean_edge > 0 else 0
        
        self.mean_edge_label.setText(f"Mean edge: {mean_edge:.4f}")
        self.uniformity_label.setText(f"Uniformity: {uniformity:.4f}")
        
        # Clear existing visualization
        while self.tri_viz_layout.count():
            item = self.tri_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Create lines from hull points if available
        hull_lines = None
        if hull_points is not None and len(hull_points) > 3:
            hull_lines = []
            for i in range(len(hull_points)-1):
                start = hull_points[i]
                end = hull_points[i+1]
                hull_lines.append([start, end])
                
        # Display triangulation
        title = f"Triangulation: {len(triangles)} triangles, {len(vertices)} vertices"
        
        # Use matplotlib or PyVista based on availability
        if self.view_3d_enabled:
            self._create_3d_visualization(
                self.tri_viz_frame,
                vertices,
                title=title,
                lines=hull_lines,
                triangles=triangles
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot triangulation using triplot
            from matplotlib.tri import Triangulation
            tri = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
            ax.triplot(tri, 'b-', lw=0.5, alpha=0.7)
            
            # Plot hull boundary if available
            if hull_points is not None and len(hull_points) > 3:
                ax.plot(hull_points[:, 0], hull_points[:, 1], 'r-', linewidth=2)
            
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Set limits with some margin
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            
            # Add 10% margin
            margin = 0.1 * np.max(max_coords - min_coords)
            
            ax.set_xlim(min_coords[0] - margin, max_coords[0] + margin)
            ax.set_ylim(min_coords[1] - margin, max_coords[1] + margin)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.tri_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.tri_viz_frame)
            self.tri_viz_layout.addWidget(toolbar)
    
    def export_results(self):
        """Export the triangulation results to a file"""
        self.statusBar().showMessage("Exporting results...")
        
        if self.triangulation_result is None:
            self.statusBar().showMessage("Error: No triangulation results to export")
            QMessageBox.critical(self, "Error", "No triangulation results to export")
            return
        
        # Create export options dialog
        export_dialog = QDialog(self)
        export_dialog.setWindowTitle("Export Options")
        export_dialog.setMinimumWidth(400)
        dialog_layout = QVBoxLayout(export_dialog)
        
        # Export format frame
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout(format_group)
        
        format_radio_group = QButtonGroup(export_dialog)
        
        obj_radio = QRadioButton("OBJ Format (3D objects)")
        ply_radio = QRadioButton("PLY Format (3D mesh)")
        stl_radio = QRadioButton("STL Format (3D printing)")
        csv_radio = QRadioButton("CSV Format (raw data)")
        
        format_radio_group.addButton(obj_radio, 0)
        format_radio_group.addButton(ply_radio, 1)
        format_radio_group.addButton(stl_radio, 2)
        format_radio_group.addButton(csv_radio, 3)
        
        obj_radio.setChecked(True)  # Default selection
        
        format_layout.addWidget(obj_radio)
        format_layout.addWidget(ply_radio)
        format_layout.addWidget(stl_radio)
        format_layout.addWidget(csv_radio)
        
        dialog_layout.addWidget(format_group)
        
        # 3D export options (enabled only if PyVista is available)
        options_group = QGroupBox("3D Export Options")
        options_layout = QGridLayout(options_group)
        
        export_3d_checkbox = QCheckBox("Export as 3D Surface")
        export_3d_checkbox.setChecked(HAVE_PYVISTA)
        export_3d_checkbox.setEnabled(HAVE_PYVISTA)
        options_layout.addWidget(export_3d_checkbox, 0, 0, 1, 2)
        
        options_layout.addWidget(QLabel("Height Scale:"), 1, 0)
        height_scale_slider = QSlider(Qt.Horizontal)
        height_scale_slider.setMinimum(0)
        height_scale_slider.setMaximum(100)
        height_scale_slider.setValue(int(self.height_factor * 100))
        options_layout.addWidget(height_scale_slider, 1, 1)
        
        dialog_layout.addWidget(options_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        export_btn = QPushButton("Export")
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(export_btn)
        
        dialog_layout.addLayout(button_layout)
        
        # Connect signals
        cancel_btn.clicked.connect(export_dialog.reject)
        export_btn.clicked.connect(export_dialog.accept)
        
        # Show dialog and get result
        result = export_dialog.exec_()
        
        if result == QDialog.Accepted:
            # Get selected format
            format_id = format_radio_group.checkedId()
            formats = ["obj", "ply", "stl", "csv"]
            if format_id >= 0 and format_id < len(formats):
                export_format = formats[format_id]
            else:
                export_format = "obj"  # Default
                
            # Get 3D options
            export_3d = export_3d_checkbox.isChecked() and HAVE_PYVISTA
            height_scale = height_scale_slider.value() / 100.0
            
            # Create a file save dialog
            file_filter = ""
            if export_format == "obj":
                file_filter = "OBJ files (*.obj);;All files (*.*)"
            elif export_format == "ply":
                file_filter = "PLY files (*.ply);;All files (*.*)"
            elif export_format == "stl":
                file_filter = "STL files (*.stl);;All files (*.*)"
            elif export_format == "csv":
                file_filter = "CSV files (*.csv);;All files (*.*)"
                
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save triangulation results",
                "",
                file_filter,
                options=QFileDialog.Options()
            )
            
            if not file_path:
                self.statusBar().showMessage("Export canceled")
                return
            
            try:
                vertices = self.triangulation_result['vertices']
                triangles = self.triangulation_result['triangles']
                
                # If 3D export is selected and PyVista is available
                if export_3d and HAVE_PYVISTA:
                    try:
                        # Create 3D vertices with Z-values
                        points_3d = np.zeros((len(vertices), 3))
                        points_3d[:, 0] = vertices[:, 0]  # X remains the same
                        points_3d[:, 1] = vertices[:, 1]  # Y remains the same
                        
                        # Calculate Z values (same as in show_3d_view)
                        x_scale = 0.05
                        y_scale = 0.05
                        z_scale = 20.0 * height_scale
                        
                        for i in range(len(vertices)):
                            x, y = vertices[i]
                            z = np.sin(x * x_scale) * np.cos(y * y_scale) * z_scale
                            points_3d[i, 2] = z
                        
                        # Create PyVista mesh
                        cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
                        mesh = pv.PolyData(points_3d, cells)
                        
                        # Save based on format
                        if export_format == "obj":
                            mesh.save(file_path)
                            self.statusBar().showMessage(f"Exported 3D OBJ file to {file_path}")
                        elif export_format == "ply":
                            mesh.save(file_path)
                            self.statusBar().showMessage(f"Exported 3D PLY file to {file_path}")
                        elif export_format == "stl":
                            mesh.save(file_path)
                            self.statusBar().showMessage(f"Exported 3D STL file to {file_path}")
                        elif export_format == "csv":
                            # For CSV, fall back to standard CSV export
                            self._export_csv(file_path, points_3d, triangles)
                            self.statusBar().showMessage(f"Exported 3D CSV file to {file_path}")
                    except Exception as e:
                        self.statusBar().showMessage(f"Error creating 3D export: {str(e)}")
                        logger.error(f"Error creating 3D export: {str(e)}")
                        QMessageBox.critical(self, "Export Error", f"Error creating 3D export: {str(e)}")
                else:
                    # Standard 2D format exports
                    if export_format == "obj":
                        self._export_obj(file_path, vertices, triangles)
                        self.statusBar().showMessage(f"Exported 2D OBJ file to {file_path}")
                    elif export_format == "ply":
                        self._export_ply(file_path, vertices, triangles)
                        self.statusBar().showMessage(f"Exported 2D PLY file to {file_path}")
                    elif export_format == "stl":
                        self._export_stl(file_path, vertices, triangles)
                        self.statusBar().showMessage(f"Exported 2D STL file to {file_path}")
                    elif export_format == "csv":
                        self._export_csv(file_path, vertices, triangles)
                        self.statusBar().showMessage(f"Exported 2D CSV file to {file_path}")
                
            except Exception as e:
                self.statusBar().showMessage(f"Error exporting results: {str(e)}")
                logger.error(f"Error exporting results: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
        else:
            self.statusBar().showMessage("Export canceled")
    
    def _export_obj(self, file_path, vertices, triangles):
        """Export to OBJ format"""
        with open(file_path, 'w') as f:
            # Write header
            f.write("# OBJ file created by MeshIt Workflow GUI\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Triangles: {len(triangles)}\n\n")
            
            # Write vertices
            for vertex in vertices:
                # OBJ indices start from 1, not 0
                f.write(f"v {vertex[0]} {vertex[1]} 0.0\n")
            
            # Write triangles
            for triangle in triangles:
                # OBJ indices start from 1, not 0
                f.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")
    
    def _export_ply(self, file_path, vertices, triangles):
        """Export to PLY format"""
        with open(file_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(triangles)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} 0.0\n")
            
            # Write triangles
            for triangle in triangles:
                f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")
    
    def _export_stl(self, file_path, vertices, triangles):
        """Export to STL format"""
        with open(file_path, 'w') as f:
            # Write header
            f.write("solid meshit_triangulation\n")
            
            # Write triangles
            for triangle in triangles:
                # Get vertices of this triangle
                v1 = vertices[triangle[0]]
                v2 = vertices[triangle[1]]
                v3 = vertices[triangle[2]]
                
                # Calculate normal (cross product of two sides)
                # For a flat mesh, all normals will be (0,0,1)
                f.write(f"  facet normal 0.0 0.0 1.0\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {v1[0]} {v1[1]} 0.0\n")
                f.write(f"      vertex {v2[0]} {v2[1]} 0.0\n")
                f.write(f"      vertex {v3[0]} {v3[1]} 0.0\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            # Write footer
            f.write("endsolid meshit_triangulation\n")
    
    def _export_csv(self, file_path, vertices, triangles):
        """Export to CSV format"""
        with open(file_path, 'w') as f:
            # Write header
            f.write("# MeshIt Workflow GUI triangulation results\n")
            f.write("# VERTICES\n")
            f.write("x,y,z\n")
            
            # Write vertices
            for vertex in vertices:
                if vertex.shape[0] == 3:  # 3D vertex
                    f.write(f"{vertex[0]},{vertex[1]},{vertex[2]}\n")
                else:  # 2D vertex
                    f.write(f"{vertex[0]},{vertex[1]},0.0\n")
            
            # Write separator
            f.write("\n# TRIANGLES\n")
            f.write("v1,v2,v3\n")
            
            # Write triangles
            for triangle in triangles:
                f.write(f"{triangle[0]},{triangle[1]},{triangle[2]}\n")

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeshItWorkflowGUI()
    window.show()
    sys.exit(app.exec_()) 