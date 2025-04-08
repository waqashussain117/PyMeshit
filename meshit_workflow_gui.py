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
from matplotlib.colors import ListedColormap, to_rgba
import re

# Import PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QTabWidget,
                            QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QGroupBox, QRadioButton, QSlider, QLineEdit,
                            QSplitter, QDialog, QFormLayout, QButtonGroup, QMenu, QAction,
                            QListWidget, QColorDialog, QListWidgetItem, QProgressDialog,
                            QActionGroup)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QObject, QThread, QTimer # Add QTimer
from PyQt5.QtGui import QFont, QIcon, QColor

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

# Worker class for background computations
class ComputationWorker(QObject):
    dataset_finished = pyqtSignal(int, str, bool) # index, name, success
    batch_finished = pyqtSignal(int, int, float) # success_count, total_eligible, elapsed_time
    error_occurred = pyqtSignal(str)

    def __init__(self, gui_instance):
        super().__init__()
        self.gui = gui_instance # Keep a reference to the GUI instance to call its methods
        self._is_running = True

    def stop(self):
        logger.debug("ComputationWorker stop requested.")
        self._is_running = False

    def compute_hulls_batch(self):
        """Worker method to compute hulls for all datasets."""
        logger.info("Worker: Starting hull computation batch.")
        if not self.gui.datasets:
            self.error_occurred.emit("No datasets loaded.")
            self.batch_finished.emit(0, 0, 0)
            logger.info("Worker: Hull batch finished (no datasets).")
            return

        success_count = 0
        total_datasets = len(self.gui.datasets)
        start_time = time.time()

        for i in range(total_datasets):
            if not self._is_running:
                logger.info(f"Worker: Hull computation loop canceled at index {i}.")
                break
            dataset_name = self.gui.datasets[i].get('name', f"Dataset {i}")
            logger.debug(f"Worker: Computing hull for '{dataset_name}' (index {i}).")
            # Moved the check inside the loop iteration, just before the potentially long call
            if not self._is_running:
                 logger.info(f"Worker: Hull computation canceled just before processing index {i}.")
                 break
            success = self.gui._compute_hull_for_dataset(i) # Call GUI's compute method
            logger.debug(f"Worker: Hull computation for index {i} finished (Success: {success}).")
            self.dataset_finished.emit(i, dataset_name, success)
            if success:
                success_count += 1

        elapsed = time.time() - start_time
        logger.info(f"Worker: Hull batch finished. Success: {success_count}/{total_datasets}. Elapsed: {elapsed:.2f}s.")
        self.batch_finished.emit(success_count, total_datasets, elapsed)
        # self._is_running = False # Resetting here might be redundant if worker is deleted

    def compute_segments_batch(self):
        """Worker method to compute segments for all eligible datasets."""
        logger.info("Worker: Starting segment computation batch.")
        datasets_with_hulls_indices = [i for i, d in enumerate(self.gui.datasets) if d.get('hull_points') is not None]

        if not datasets_with_hulls_indices:
            self.error_occurred.emit("No datasets have computed hulls.")
            self.batch_finished.emit(0, 0, 0)
            logger.info("Worker: Segment batch finished (no eligible datasets).")
            return

        success_count = 0
        total_eligible = len(datasets_with_hulls_indices)
        start_time = time.time()

        for i in datasets_with_hulls_indices:
            if not self._is_running:
                 logger.info(f"Worker: Segment computation loop canceled at index {i}.")
                 break
            dataset_name = self.gui.datasets[i].get('name', f"Dataset {i}")
            logger.debug(f"Worker: Computing segments for '{dataset_name}' (index {i}).")
             # Moved the check inside the loop iteration
            if not self._is_running:
                 logger.info(f"Worker: Segment computation canceled just before processing index {i}.")
                 break
            success = self.gui._compute_segments_for_dataset(i) # Call GUI's compute method
            logger.debug(f"Worker: Segment computation for index {i} finished (Success: {success}).")
            self.dataset_finished.emit(i, dataset_name, success)
            if success:
                success_count += 1

        elapsed = time.time() - start_time
        logger.info(f"Worker: Segment batch finished. Success: {success_count}/{total_eligible}. Elapsed: {elapsed:.2f}s.")
        self.batch_finished.emit(success_count, total_eligible, elapsed)
        # self._is_running = False

    def compute_triangulations_batch(self):
        """Worker method to compute triangulations for all eligible datasets."""
        logger.info("Worker: Starting triangulation computation batch.")
        datasets_with_segments_indices = [i for i, d in enumerate(self.gui.datasets) if d.get('segments') is not None]

        if not datasets_with_segments_indices:
            self.error_occurred.emit("No datasets have computed segments.")
            self.batch_finished.emit(0, 0, 0)
            logger.info("Worker: Triangulation batch finished (no eligible datasets).")
            return

        success_count = 0
        total_eligible = len(datasets_with_segments_indices)
        start_time = time.time()

        for i in datasets_with_segments_indices:
            if not self._is_running:
                 logger.info(f"Worker: Triangulation computation loop canceled at index {i}.")
                 break
            dataset_name = self.gui.datasets[i].get('name', f"Dataset {i}")
            logger.debug(f"Worker: Computing triangulation for '{dataset_name}' (index {i}).")
             # Moved the check inside the loop iteration
            if not self._is_running:
                 logger.info(f"Worker: Triangulation computation canceled just before processing index {i}.")
                 break
            success = self.gui._run_triangulation_for_dataset(i) # Call GUI's compute method
            logger.debug(f"Worker: Triangulation computation for index {i} finished (Success: {success}).")
            self.dataset_finished.emit(i, dataset_name, success)
            if success:
                success_count += 1

        elapsed = time.time() - start_time
        logger.info(f"Worker: Triangulation batch finished. Success: {success_count}/{total_eligible}. Elapsed: {elapsed:.2f}s.")
        self.batch_finished.emit(success_count, total_eligible, elapsed)
        # self._is_running = False


class MeshItWorkflowGUI(QMainWindow):
    # Define a color cycle for datasets
    DEFAULT_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    def __init__(self):
        """Initialize the GUI"""
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("MeshIt Workflow GUI")
        self.setGeometry(100, 100, 1400, 900) # Increased default size
        
        # Initialize data structures for multiple datasets
        self.datasets = []  # List to hold dataset dictionaries
        self.current_dataset_index = -1  # Index of the currently active dataset
        self._color_index = 0 # For assigning colors to datasets
        
        # Visualization placeholders and state
        self.points = None
        self.hull_points = None
        self.segments = None
        self.triangulation_result = None
        self.active_view = "points" # To track which visualization is active
        
        # Initialize optional variables that could be used for 3D visualization
        self.view_3d_enabled = HAVE_PYVISTA
        self.height_factor = 1.0
        self.current_plotter = None
        self.pv_plotter = None
        
        # Initialize legend widgets
        self.file_legend_widget = None
        self.hull_legend_widget = None
        self.segment_legend_widget = None
        self.tri_legend_widget = None
        
        # Create layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create notebook with multiple tabs
        self.notebook = QTabWidget()
        self.main_layout.addWidget(self.notebook)
        
        # Setup file tab
        self.file_tab = QWidget()
        self.notebook.addTab(self.file_tab, "1. Load Data")
        self._setup_file_tab()
        
        # Setup hull tab
        self.hull_tab = QWidget()
        self.notebook.addTab(self.hull_tab, "2. Convex Hull")
        self._setup_hull_tab()
        
        # Setup segment tab
        self.segment_tab = QWidget()
        self.notebook.addTab(self.segment_tab, "3. Segmentation")
        self._setup_segment_tab()
        
        # Setup triangulation tab
        self.triangulation_tab = QWidget()
        self.notebook.addTab(self.triangulation_tab, "4. Triangulation")
        self._setup_triangulation_tab()
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Connect signals
        self.notebook.currentChanged.connect(self._on_tab_changed)
        
        # Show ready message
        self.statusBar().showMessage("MeshIt GUI Ready")
        
        # Initialize plot axes (will be properly set up in _setup_..._tab methods)
        self.hull_ax = None
        self.segment_ax = None
        self.tri_ax = None
        self.hull_canvas = None
        self.segment_canvas = None
        self.tri_canvas = None
        
        # For 3D view management
        self.plotter = None
        self.vtk_widget = None
        
        self._create_main_layout()
        self.show()
        self._update_statistics() # Initial update
    
    def _create_menu_bar(self):
        """Create the menu bar"""
        menu_bar = self.menuBar()
        
        # --- File Menu ---
        file_menu = menu_bar.addMenu("&File")

        load_action = QAction("&Load File...", self)
        load_action.setStatusTip("Load a single point data file")
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)

        load_multiple_action = QAction("Load &Multiple Files...", self)
        load_multiple_action.setStatusTip("Load multiple point data files as separate datasets")
        load_multiple_action.triggered.connect(self.load_multiple_files)
        file_menu.addAction(load_multiple_action)

        generate_action = QAction("&Generate Test Data...", self)
        generate_action.setStatusTip("Generate sample point data for testing")
        generate_action.triggered.connect(self.generate_test_data)
        file_menu.addAction(generate_action)
        
        file_menu.addSeparator()

        export_action = QAction("&Export Active Result...", self)
        export_action.setStatusTip("Export the triangulation result of the active dataset")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Dataset Menu ---
        dataset_menu = menu_bar.addMenu("&Datasets")

        process_all_action = dataset_menu.addAction("&Process All Datasets")
        process_all_action.setStatusTip("Run all steps (hull, segments, triangulation) for all datasets")
        process_all_action.triggered.connect(self.process_all_datasets)
        
        dataset_menu.addSeparator()

        remove_active_action = dataset_menu.addAction("&Remove Active Dataset")
        remove_active_action.setStatusTip("Remove the currently selected dataset from the list")
        remove_active_action.triggered.connect(self.remove_active_dataset)

        clear_all_action = dataset_menu.addAction("Clear &All Datasets")
        clear_all_action.setStatusTip("Remove all loaded datasets")
        clear_all_action.triggered.connect(self.clear_all_datasets)


        # --- Visualization Menu ---
        viz_menu = menu_bar.addMenu("&Visualization")

        view_3d_action = viz_menu.addAction("Show &3D View")
        view_3d_action.setStatusTip("Open a separate interactive 3D view (requires PyVista)")
        view_3d_action.triggered.connect(self.show_3d_view)
        view_3d_action.setEnabled(HAVE_PYVISTA) # Disable if PyVista not available
        
        viz_menu.addSeparator()
        
        # --- Height Factor Submenu ---
        height_menu = viz_menu.addMenu("Height Factor (3D)")
        height_group = QActionGroup(self) # Use QActionGroup for radio-button like behavior

        height_factors = [0.1, 0.5, 1.0, 2.0, 5.0]
        for factor in height_factors:
            height_action = QAction(f"{factor:.1f}x", self, checkable=True)
            height_action.setActionGroup(height_group)
            height_action.triggered.connect(lambda checked, f=factor: self._set_height_factor_and_update(f) if checked else None)
            if abs(factor - self.height_factor) < 1e-6: # Check if it's the current factor
                 height_action.setChecked(True)
            height_menu.addAction(height_action)


        # --- Help Menu ---
        help_menu = menu_bar.addMenu("&Help")
        about_action = help_menu.addAction("&About")
        about_action.setStatusTip("Show information about the application")
        about_action.triggered.connect(self._show_about)
    
    def _setup_file_tab(self):
        """Sets up the file loading tab with controls and visualization area"""
        # Main layout for the tab
        tab_layout = QHBoxLayout(self.file_tab)
        
        # --- Control panel (left side) ---
        control_panel = QWidget()
        control_panel.setMaximumWidth(350) # Limit width
        control_layout = QVBoxLayout(control_panel)
        
        # -- File Loading Controls --
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout(file_group)
        
        # Buttons for loading files
        load_btn = QPushButton("Load Single File...")
        load_btn.setToolTip("Load points from a single file (.txt, .csv, .pts)")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)
        
        # Button for loading multiple files
        load_multiple_btn = QPushButton("Load Multiple Files...")
        load_multiple_btn.setToolTip("Load points from multiple files as separate datasets")
        load_multiple_btn.clicked.connect(self.load_multiple_files)
        file_layout.addWidget(load_multiple_btn)
        
        # Test data generation
        test_btn = QPushButton("Generate Test Data")
        test_btn.setToolTip("Generate sample point data for testing")
        test_btn.clicked.connect(self.generate_test_data)
        file_layout.addWidget(test_btn)
        
        control_layout.addWidget(file_group)
        
        # -- Dataset List --
        datasets_group = QGroupBox("Datasets")
        datasets_layout = QVBoxLayout(datasets_group)
        
        self.dataset_list_widget = QListWidget()
        self.dataset_list_widget.setToolTip("List of loaded datasets. Select one to view/process.")
        # Enable extended selection later if needed: self.dataset_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.dataset_list_widget.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        self.dataset_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.dataset_list_widget.customContextMenuRequested.connect(self._show_dataset_context_menu)
        datasets_layout.addWidget(self.dataset_list_widget)
        
        control_layout.addWidget(datasets_group)
        
        # -- Statistics --
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Number of datasets
        self.num_datasets_label = QLabel("Datasets: 0")
        stats_layout.addWidget(self.num_datasets_label)
        
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
        compute_btn = QPushButton("Compute Convex Hull (All Datasets)")
        compute_btn.setObjectName("compute_btn") # Set the object name
        compute_btn.setToolTip("Compute convex hull for all loaded datasets")
        compute_btn.clicked.connect(self.compute_all_hulls)
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
        compute_btn = QPushButton("Compute Segmentation (All Datasets)") # Update button text
        compute_btn.setObjectName("compute_btn") # Set the object name
        compute_btn.setToolTip("Compute segmentation for all datasets with computed hulls") # Update tooltip
        compute_btn.clicked.connect(self.compute_all_segments) # Connect to the new batch method
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
        run_btn = QPushButton("Run Triangulation (All Datasets)") # Update button text
        run_btn.setObjectName("run_btn") # Set the object name
        run_btn.setToolTip("Run triangulation for all datasets with computed segments") # Update tooltip
        run_btn.clicked.connect(self.run_all_triangulations) # Connect to the new batch method
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
            # First check file content to see if it might be a fault file format
            try:
                with open(file_path, 'r') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                    has_mixed_delimiters = any(('\t' in line and (',' in line or ';' in line)) for line in first_lines if line)
                    
                if has_mixed_delimiters or any(line.startswith("Bounds:") for line in first_lines if line):
                    # Likely a fault file with mixed delimiters, use custom parser
                    points = self._parse_fault_file(file_path)
                    if points is not None and len(points) > 0:
                        logger.info(f"Successfully parsed file as fault format: {len(points)} points")
                        return points
            except Exception as e:
                # If preview fails, continue with standard import
                logger.debug(f"File preview check failed: {str(e)}")
                
            # Try different formats based on file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Check if this might be a fault file with mixed delimiters
            if ext in ['.fault', '.txt', '.dat', '']:
                try:
                    # Custom parsing for fault files with mixed delimiters
                    points = self._parse_fault_file(file_path)
                    if points is not None and len(points) > 0:
                        return points
                except Exception as e:
                    logger.debug(f"Fault file parsing failed, trying standard formats: {str(e)}")
                    # Continue with standard formats if fault parsing fails
            
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
            
            # If data has only 2 columns (X, Y), add a Z column with zeros
            if points.shape[1] == 2:
                points_3d = np.zeros((len(points), 3))
                points_3d[:, 0:2] = points
                return points_3d
            
            # If data has 3 or more columns, use the first 3 as X, Y, Z
            return points[:, 0:3]
            
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error reading file: {str(e)}")
            return None
            
    def _extract_all_values_from_line(self, line):
        """
        Extract all numeric values from a line with mixed delimiters
        
        Args:
            line: String line to parse
            
        Returns:
            List of numeric values
        """
        # First, replace tabs with spaces to ensure they're proper separators
        line = line.replace('\t', ' ')
        
        # Replace '+' with ' +' and '-' with ' -' to ensure they're separated
        # But don't add spaces if they're after digits (like in 1.5e-3)
        line = re.sub(r'(?<!\d)(\+|-)', r' \1', line)
        
        # Replace all common delimiters with spaces
        for delimiter in [',', ';']:
            line = line.replace(delimiter, ' ')
        
        # Remove extra spaces
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Split by whitespace
        parts = line.split()
        
        # Convert to float
        values = []
        for part in parts:
            try:
                # Remove '+' sign prefix if present
                if part.startswith('+'):
                    part = part[1:]
                values.append(float(part))
            except ValueError:
                logger.debug(f"Could not convert '{part}' to float in line: '{line}'")
                continue
        
        # Debug output
        logger.debug(f"Extracted values {values} from line: '{line}'")
        return values
    
    def _parse_fault_file(self, file_path):
        """
        Parse a fault file with mixed delimiters 
        
        Args:
            file_path: Path to the fault file
            
        Returns:
            numpy array of points (Nx3) or None if parsing failed
        """
        points = []
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue  # Skip empty lines and comments
                    
                    # Check if this is a bounds format line: "Bounds: X[min,max] Y[min,max] Z[min,max]"
                    if line.startswith("Bounds:"):
                        bounds_points = self._parse_bounds_format(line)
                        if bounds_points is not None and len(bounds_points) > 0:
                            points.extend(bounds_points)
                        continue
                    
                    # Parse each line as a single (x,y,z) coordinate
                    try:
                        # Extract all numbers from the line
                        all_values = self._extract_all_values_from_line(line)
                        
                        # We need at least 2 values for a valid point
                        if len(all_values) >= 2:
                            x = all_values[0]
                            y = all_values[1]
                            # Use third value as z if available, otherwise default to 0
                            z = all_values[2] if len(all_values) >= 3 else 0.0
                            points.append([x, y, z])
                            logger.debug(f"Parsed point: ({x}, {y}, {z})")
                    except Exception as e:
                        logger.debug(f"Error parsing line {line_num+1}: {str(e)}")
            
            if len(points) == 0:
                return None
                
            return np.array(points)
        except Exception as e:
            logger.debug(f"Error in fault file parsing: {str(e)}")
            return None
    
    def _parse_bounds_format(self, line):
        """
        Parse bounds format line: "Bounds: X[min,max] Y[min,max] Z[min,max]"
        
        Args:
            line: String line in bounds format
            
        Returns:
            List of points representing the bounds
        """
        try:
            # Extract X, Y, Z ranges
            x_range = re.search(r'X\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]', line)
            y_range = re.search(r'Y\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]', line)
            z_range = re.search(r'Z\[([-+]?\d+\.?\d*),([-+]?\d+\.?\d*)\]', line)
            
            if not x_range or not y_range:
                logger.debug("Could not find X or Y ranges in bounds line")
                return None
                
            # Extract min/max values for each dimension
            x_min, x_max = float(x_range.group(1)), float(x_range.group(2))
            y_min, y_max = float(y_range.group(1)), float(y_range.group(2))
            
            # For Z, use found values or default to 0
            if z_range:
                z_min, z_max = float(z_range.group(1)), float(z_range.group(2))
            else:
                z_min, z_max = 0.0, 0.0
                
            # Create corner points only (not the full bounding box)
            # For a fault file, we typically just want the 4 corners of the rectangular region
            bounds_points = [
                [x_min, y_min, 0.0],  # Bottom-left
                [x_max, y_min, 0.0],  # Bottom-right
                [x_max, y_max, 0.0],  # Top-right
                [x_min, y_max, 0.0]   # Top-left
            ]
            
            # Log the extracted points for debugging
            logger.info(f"Parsed bounds points: {bounds_points}")
            
            return bounds_points
        except Exception as e:
            logger.debug(f"Error parsing bounds format: {str(e)}")
            return None
    
    def _extract_values(self, part):
        """
        Extract numeric values from a string that might contain multiple delimiters
        
        Args:
            part: String part potentially containing multiple values
            
        Returns:
            List of numeric values
        """
        values = []
        
        # Replace '+' at the beginning of numbers with nothing
        part = re.sub(r'(?<!\d)\+', '', part)
        
        # First try comma
        comma_parts = part.split(',')
        for cp in comma_parts:
            # Then try semicolon
            semicolon_parts = cp.split(';')
            for sp in semicolon_parts:
                # Clean and convert to float
                sp = sp.strip()
                if sp:
                    try:
                        values.append(float(sp))
                    except ValueError:
                        logger.debug(f"Could not convert '{sp}' to float")
        
        return values

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
            points: 2D or 3D points to visualize
        """
        if points is None or len(points) == 0:
            return
            
        # Update statistics
        self.points_label.setText(f"Points: {len(points)}")
        
        # Calculate bounds, correctly handling 3D coordinates
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        min_x, min_y = min_coords[0], min_coords[1]
        max_x, max_y = max_coords[0], max_coords[1]
        
        # If 3D data is available, also show Z range
        if points.shape[1] >= 3:
            min_z, max_z = min_coords[2], max_coords[2]
            self.bounds_label.setText(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}] Z[{min_z:.2f},{max_z:.2f}]")
        else:
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
    
    def _compute_hull_for_dataset(self, dataset_index):
        """Compute the convex hull for a specific dataset index. Returns True on success, False on error."""
        # Check index validity
        if not (0 <= dataset_index < len(self.datasets)):
            logger.error(f"Invalid dataset index {dataset_index} for hull computation.")
            return False

        dataset = self.datasets[dataset_index]
        dataset_name = dataset.get('name', f"Dataset {dataset_index}")
        # self.statusBar().showMessage(f"Computing hull for {dataset_name}...") # Remove status update
        
        if dataset.get('points') is None or len(dataset['points']) < 3:
            self.statusBar().showMessage(f"Error: Need at least 3 points for hull in {dataset_name}")
            logger.warning(f"Skipping hull for {dataset_name}: needs >= 3 points.")
            return False # Indicate error or skip
        
        try:
            # Get points
            points = dataset['points']
            
            # For 3D computations, we need to handle the hull differently
            if points.shape[1] >= 3:
                # Check if points are roughly planar (most real-world datasets are)
                # 1. Center points
                centroid = np.mean(points, axis=0)
                centered = points - centroid
                
                # 2. Get dominant plane using PCA/SVD
                u, s, vh = np.linalg.svd(centered, full_matrices=False)
                # The smallest singular value/component tells us how planar the data is
                normal = vh[2]  # This is the approximate normal to the best-fitting plane
                
                # For highly planar data (ratio of smallest to second smallest singular value is small)
                if s[2] / s[1] < 0.2:  # Threshold can be adjusted
                    logger.debug(f"Points in {dataset_name} are mostly planar, using projected hull")
                    # Project points onto a plane defined by first two principal components
                    projected = np.dot(centered, vh[:2].T)
                    
                    # Compute 2D convex hull on projected points
                    from scipy.spatial import ConvexHull
                    hull_2d = ConvexHull(projected)
                    
                    # Map hull vertices back to 3D
                    hull_indices = hull_2d.vertices
                    hull_points = points[hull_indices]
                    
                    # Ensure the hull is closed
                    if not np.array_equal(hull_points[0], hull_points[-1]):
                        hull_points = np.vstack([hull_points, hull_points[0]])
                else:
                    # For truly 3D point clouds, use 3D convex hull
                    logger.debug(f"Points in {dataset_name} are 3D, using 3D convex hull")
                    from scipy.spatial import ConvexHull
                    hull_3d = ConvexHull(points)
                    
                    # Extract unique vertices from the hull
                    # 3D hulls use triangular faces, so we need to get unique vertices
                    hull_indices = []
                    for simplex in hull_3d.simplices:
                        for idx in simplex:
                            if idx not in hull_indices:
                                hull_indices.append(idx)
                    
                    # Extract hull points in original order
                    hull_points = points[hull_indices]
                    
                    # For visualization purposes, we'll close the hull by adding the first point again
                    hull_points = np.vstack([hull_points, hull_points[0]])
            else:
                # Standard 2D hull computation
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points)
                
                # Extract hull points
                hull_points = points[hull.vertices]
                
                # Close the hull
                hull_points = np.append(hull_points, [hull_points[0]], axis=0)
            
            # Store in dataset
            dataset['hull_points'] = hull_points
            
            # Clear any previous results from later steps for this dataset
            dataset.pop('segments', None)
            dataset.pop('triangulation_result', None)
            
            return True # Indicate success
            
        except Exception as e:
            # self.statusBar().showMessage(f"Error computing hull for {dataset_name}: {str(e)}") # Remove status update
            logger.error(f"Error computing hull for {dataset_name}: {str(e)}")
            # Optionally show message box for individual errors during batch?
            # QMessageBox.critical(self, "Hull Error", f"Error computing hull for {dataset_name}: {str(e)}")
            return False # Indicate error

    def compute_hull(self):
        """Compute the convex hull of the loaded points for the *active* dataset (primarily for context menu)"""
        # Check if we have an active dataset
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.datasets):
            self.statusBar().showMessage("Error: No active dataset selected")
            QMessageBox.critical(self, "Error", "No active dataset selected")
            return

        self.statusBar().showMessage(f"Computing hull for {self.datasets[self.current_dataset_index]['name']}...") # Add status message here for single run
        success = self._compute_hull_for_dataset(self.current_dataset_index)

        if success:
            # Update statistics and visualization after computing for the active one
            self._update_statistics()
            self._visualize_all_hulls()
            self._update_visualization() # Ensure other dependent views are updated/cleared
            self.notebook.setCurrentIndex(1)  # Switch to hull tab
            self.statusBar().showMessage(f"Computed hull for {self.datasets[self.current_dataset_index]['name']}") # Add success message here

    def compute_all_hulls(self):
        """Compute the convex hull for all loaded datasets using a worker thread."""
        if not self.datasets:
            QMessageBox.information(self, "No Data", "No datasets loaded to compute hulls.")
            return
        self._run_batch_computation("hulls", len(self.datasets))
    
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
            # Extract x,y coordinates only
            x1, y1 = hull_points[i][0], hull_points[i][1]
            x2, y2 = hull_points[i+1][0], hull_points[i+1][1]
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
        
        # Ensure points are in 3D format
        if points.shape[1] == 2:
            # Convert 2D points to 3D with Z=0
            points_3d = np.zeros((len(points), 3))
            points_3d[:, 0] = points[:, 0]
            points_3d[:, 1] = points[:, 1]
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
                        line_points[0, 0:2] = line[0]
                        line_points[1, 0:2] = line[1]
                        
                        # Set Z values from the original points if possible
                        # Find closest point in the points array for each line endpoint
                        for j in range(2):
                            # Find the point in the dataset closest to this line endpoint
                            distances = np.sum((points_3d[:, 0:2] - line[j])**2, axis=1)
                            closest_idx = np.argmin(distances)
                            line_points[j, 2] = points_3d[closest_idx, 2]
                        
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
                    # Convert feature points to 3D if needed
                    if feature_points.shape[1] == 2:
                        fp_3d = np.zeros((len(feature_points), 3))
                        fp_3d[:, 0:2] = feature_points
                        
                        # Set Z values from closest points in the main point cloud
                        for i in range(len(feature_points)):
                            point_2d = feature_points[i]
                            distances = np.sum((points_3d[:, 0:2] - point_2d)**2, axis=1)
                            closest_idx = np.argmin(distances)
                            fp_3d[i, 2] = points_3d[closest_idx, 2]
                    else:
                        fp_3d = feature_points.copy()
                    
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
        # Store the new height factor
        self.height_factor = height_factor
        
        # The height factor is used to exaggerate the Z dimension for better visualization
        # If we have a current plotter, update it
        if hasattr(self, 'current_plotter') and self.current_plotter is not None:
            try:
                # Scale the Z dimension using plotter's scale feature
                # This adjusts how the 3D scene appears without modifying the actual coordinates
                self.current_plotter.set_scale(zscale=height_factor)
                self.current_plotter.reset_camera()
            except Exception as e:
                logger.error(f"Error updating 3D Z exaggeration: {str(e)}")
    
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
            # Convert 2D points to 3D with Z=0
            points_3d = np.zeros((len(points), 3))
            points_3d[:, 0] = points[:, 0]
            points_3d[:, 1] = points[:, 1]
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
                    line_points[0, 0:2] = line[0]
                    line_points[1, 0:2] = line[1]
                    
                    # Set Z values from the original points if possible
                    for j in range(2):
                        # Find the point in the dataset closest to this line endpoint
                        distances = np.sum((points_3d[:, 0:2] - line[j])**2, axis=1)
                        closest_idx = np.argmin(distances)
                        line_points[j, 2] = points_3d[closest_idx, 2]
                    
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
                # Convert feature points to 3D if needed
                if feature_points.shape[1] == 2:
                    fp_3d = np.zeros((len(feature_points), 3))
                    fp_3d[:, 0:2] = feature_points
                    
                    # Set Z values from closest points in the main point cloud
                    for i in range(len(feature_points)):
                        point_2d = feature_points[i]
                        distances = np.sum((points_3d[:, 0:2] - point_2d)**2, axis=1)
                        closest_idx = np.argmin(distances)
                        fp_3d[i, 2] = points_3d[closest_idx, 2]
                else:
                    fp_3d = feature_points.copy()
                
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
        """Show 3D view of all triangulations in a separate window"""
        if not self.view_3d_enabled:
            QMessageBox.information(self, "PyVista Not Available", 
                                 "PyVista is not available. Please install PyVista for 3D visualization.")
            return
            
        # Check if we have any triangulation results
        visible_datasets = [d for d in self.datasets if d.get('visible', True) and d.get('triangulation_result') is not None]
        
        if not visible_datasets:
            QMessageBox.information(self, "No Triangulation", 
                                 "No triangulation results available. Please run triangulation first.")
            return
            
        # Close previous plotter if it exists
        if hasattr(self, 'pv_plotter') and self.pv_plotter is not None:
            try:
                self.pv_plotter.close()
            except:
                pass
        
        # Create a new plotter
        self.pv_plotter = pv.Plotter(window_size=[800, 600], title="MeshIt 3D Visualization")
        self.pv_plotter.set_background("#383F51")  # Dark blue background
        
        # Show each dataset
        for dataset in visible_datasets:
            color = dataset.get('color', '#000000')
            name = dataset.get('name', 'Unnamed')
            
            triangulation_result = dataset['triangulation_result']
            vertices = triangulation_result['vertices']
            triangles = triangulation_result['triangles']
            
            # Ensure vertices are 3D
            vertices_3d = np.zeros((len(vertices), 3))
            if vertices.shape[1] == 2:
                # If 2D vertices, try to get Z from original points
                vertices_3d[:, 0:2] = vertices
                
                # If the dataset has 3D points, use them to set Z values based on XY proximity
                original_points = dataset.get('points')
                if original_points is not None and original_points.shape[1] >= 3:
                    for i in range(len(vertices)):
                        vertex_2d = vertices[i]
                        # Find the closest original point
                        distances = np.sum((original_points[:, 0:2] - vertex_2d)**2, axis=1)
                        closest_idx = np.argmin(distances)
                        vertices_3d[i, 2] = original_points[closest_idx, 2]
            else:
                # If vertices already have Z, use them directly
                vertices_3d = vertices.copy()
            
            # Create a mesh
            cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
            mesh = pv.PolyData(vertices_3d, cells)
            
            # Add mesh to plotter
            self.pv_plotter.add_mesh(mesh, color=color, opacity=0.8, 
                                  show_edges=True, edge_color=color, 
                                  specular=0.5, smooth_shading=True,
                                  name=name)  # Add name to identify in plotter
        
        # Add text with dataset information
        text_lines = []
        for i, dataset in enumerate(visible_datasets):
            name = dataset.get('name', 'Unnamed')
            num_vertices = len(dataset['triangulation_result']['vertices'])
            num_triangles = len(dataset['triangulation_result']['triangles'])
            text_lines.append(f"{i+1}. {name}: {num_vertices} vertices, {num_triangles} triangles")
        
        if text_lines:
            info_text = '\n'.join(text_lines)
            self.pv_plotter.add_text(info_text, font_size=10, position='upper_left')
        
        # Add UI controls for adjusting display
        self.pv_plotter.add_checkbox_button_widget(
            lambda state: [self.pv_plotter.add_axes() if state else self.pv_plotter.remove_axes()],
            value=True,
            position=(10, 180),
            size=30,
            border_size=1,
            color_on='white',
            color_off='grey',
            background_color='darkblue'
        )
        self.pv_plotter.add_text("Axes", position=(50, 180), font_size=10, color='white')
        
        # Add scale slider for height
        def update_height(value):
            self._set_height_factor(value)
            self.show_3d_view()  # Refresh the view
            
        # Add axes for reference
        self.pv_plotter.add_axes()
        
        # Show the plotter in a separate window
        self.pv_plotter.show()
    
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

    def _compute_segments_for_dataset(self, dataset_index):
        """Compute the segmentation for a specific dataset index. Returns True on success, False on error."""
        # Check index validity
        if not (0 <= dataset_index < len(self.datasets)):
            logger.error(f"Invalid dataset index {dataset_index} for segmentation.")
            return False

        dataset = self.datasets[dataset_index]
        dataset_name = dataset.get('name', f"Dataset {dataset_index}")
        
        if dataset.get('hull_points') is None or len(dataset['hull_points']) < 4:  # 3 vertices + 1 closing point
            self.statusBar().showMessage(f"Skipping segments for {dataset_name}: Compute convex hull first")
            logger.warning(f"Skipping segments for {dataset_name}: hull not computed.")
            return False # Indicate error or skip

        # Get the segmentation parameters from the UI (use consistent settings for all)
        try:
            segment_length = float(self.segment_length_input.text())
            if segment_length <= 0:
                segment_length = 1.0
        except ValueError:
            segment_length = 1.0

        density_factor = self.segment_density_slider.value() / 100.0
        effective_segment_length = segment_length / density_factor
        
        try:
            # Extract the hull boundary (excluding the closing point)
            hull_boundary = dataset['hull_points'][:-1]
            hull_size = len(hull_boundary)
            
            # Performance optimization: Limit the number of segments per edge to prevent excessive calculations
            MAX_SEGMENTS_PER_EDGE = 20
            
            # Estimate average edge length to determine a reasonable minimum segment length
            total_perimeter = 0
            for i in range(hull_size):
                p1 = hull_boundary[i]
                p2 = hull_boundary[(i + 1) % hull_size]
                total_perimeter += np.linalg.norm(p2 - p1)
            
            avg_edge_length = total_perimeter / hull_size
            min_segment_length = avg_edge_length / MAX_SEGMENTS_PER_EDGE
            
            # Use the larger of calculated min_segment_length and effective_segment_length
            effective_segment_length = max(effective_segment_length, min_segment_length)
            
            # Pre-allocate segments list with reasonable capacity
            estimated_segments = int(total_perimeter / effective_segment_length) + hull_size
            segments = []
            
            # Generate segments along the hull boundary with uniform distribution
            for i in range(hull_size):
                p1 = hull_boundary[i]
                p2 = hull_boundary[(i + 1) % hull_size]
                
                # Compute edge length
                dist = np.linalg.norm(p2 - p1)
                
                # Limit number of segments per edge
                num_segments = min(max(1, int(np.ceil(dist / effective_segment_length))), MAX_SEGMENTS_PER_EDGE)
                
                # Vectorized segment creation for this edge
                t_values = np.linspace(0, 1, num_segments + 1)
                edge_segments = []
                
                for j in range(num_segments):
                    t1, t2 = t_values[j], t_values[j+1]
                    segment_start = p1 + t1 * (p2 - p1)
                    segment_end = p1 + t2 * (p2 - p1)
                    segments.append([segment_start, segment_end])
            
            # Store the segments in the dataset - using a normal list for better performance
            dataset['segments'] = segments
            
            # Clear any previous results from later steps for this dataset
            dataset.pop('triangulation_result', None)

            return True # Indicate success

        except Exception as e:
            logger.error(f"Error computing segments for {dataset_name}: {str(e)}")
            return False # Indicate error

    def compute_segments(self):
        """Compute the segmentation of the convex hull for the *active* dataset (primarily for context menu)"""
        # Check if we have an active dataset
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.datasets):
            self.statusBar().showMessage("Error: No active dataset selected")
            QMessageBox.critical(self, "Error", "No active dataset selected")
            return

        self.statusBar().showMessage(f"Computing segments for {self.datasets[self.current_dataset_index]['name']}...") # Add status message here for single run
        success = self._compute_segments_for_dataset(self.current_dataset_index)

        if success:
            # Update statistics and visualization after computing for the active one
            self._update_statistics()
            self._visualize_all_segments()
            self._update_visualization() # Ensure other dependent views are updated/cleared
            self.notebook.setCurrentIndex(2)  # Switch to segment tab
            self.statusBar().showMessage(f"Computed segments for {self.datasets[self.current_dataset_index]['name']}") # Add success message here

    def compute_all_segments(self):
        """Compute segmentation for all datasets that have hulls using a worker thread."""
        datasets_with_hulls_indices = [i for i, d in enumerate(self.datasets) if d.get('hull_points') is not None]
        if not datasets_with_hulls_indices:
            QMessageBox.information(self, "No Hulls", "No datasets have computed hulls. Please compute hulls first.")
            return
        self._run_batch_computation("segments", len(datasets_with_hulls_indices))
    
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
        segment_lengths = []
        for segment in segments:
            # Get segment endpoints correctly for either format
            if hasattr(segment[0], 'shape'):
                # NumPy array format
                p1, p2 = segment[0], segment[1]
            else:
                # List format
                p1 = np.array(segment[0])
                p2 = np.array(segment[1])
            
            segment_lengths.append(np.linalg.norm(p2 - p1))
            
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
            # Convert segments to consistent format for visualization
            segment_points = []
            for segment in segments:
                if hasattr(segment[0], 'shape'):
                    # NumPy array format
                    segment_points.append(segment[0])
                    segment_points.append(segment[1])
                else:
                    # List format
                    segment_points.append(np.array(segment[0]))
                    segment_points.append(np.array(segment[1]))
            
            # Convert to numpy array for processing
            segment_points = np.vstack(segment_points)
            
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
    
    def _run_triangulation_for_dataset(self, dataset_index):
        """Run triangulation for a specific dataset index. Returns True on success, False on error."""
        # Check index validity
        if not (0 <= dataset_index < len(self.datasets)):
            logger.error(f"Invalid dataset index {dataset_index} for triangulation.")
            return False

        dataset = self.datasets[dataset_index]
        dataset_name = dataset.get('name', f"Dataset {dataset_index}")
        
        if dataset.get('segments') is None or len(dataset['segments']) < 3:
            self.statusBar().showMessage(f"Skipping triangulation for {dataset_name}: Compute segments first")
            logger.warning(f"Skipping triangulation for {dataset_name}: segments not computed.")
            return False # Indicate error or skip

        # Get triangulation parameters from UI (use consistent settings for all)
        gradient = self.gradient_input.value()
        min_angle = self.min_angle_input.value()
        base_size_factor = self.base_size_factor_input.value()
        uniform = self.uniform_checkbox.isChecked()
        
        try:
            start_time = time.time()
            
            # Create boundary from hull points
            boundary_points = dataset['hull_points'][:-1]  # Exclude closing point
            
            # For 3D points, we need to project to a plane for triangulation
            if boundary_points.shape[1] > 2:
                logger.info(f"Projecting 3D points to best-fit plane for triangulation for {dataset_name}")
                
                # Find best-fitting plane using PCA/SVD
                # 1. Center points
                centroid = np.mean(boundary_points, axis=0)
                centered = boundary_points - centroid
                
                # 2. Get dominant plane using PCA/SVD
                u, s, vh = np.linalg.svd(centered, full_matrices=False)
                
                # 3. Use first two principal components as projection basis
                projection_basis = vh[:2]
                
                # 4. Project points to 2D plane
                points_2d = np.dot(centered, projection_basis.T)
                
                # Store original points and projection info for reconstruction
                original_boundary_points = boundary_points.copy()
                boundary_points = points_2d
                
                # Do the same for all segments to ensure consistency
                segments_2d = []
                for segment in dataset['segments']:
                    if hasattr(segment[0], 'shape') and segment[0].shape[0] > 2:
                        # For numpy arrays
                        start_centered = segment[0] - centroid
                        end_centered = segment[1] - centroid
                        
                        start_2d = np.dot(start_centered, projection_basis.T)
                        end_2d = np.dot(end_centered, projection_basis.T)
                        
                        segments_2d.append([start_2d, end_2d])
                    elif len(segment[0]) > 2:
                        # For list based points
                        start_centered = np.array(segment[0]) - centroid
                        end_centered = np.array(segment[1]) - centroid
                        
                        start_2d = np.dot(start_centered, projection_basis.T)
                        end_2d = np.dot(end_centered, projection_basis.T)
                        
                        segments_2d.append([start_2d, end_2d])
                    else:
                        # Already 2D
                        segments_2d.append(segment)
                
                dataset['segments_2d'] = segments_2d
                
                # Store projection parameters for reconstruction
                dataset['projection_params'] = {
                    'centroid': centroid,
                    'basis': projection_basis
                }
            else:
                # Store info showing no projection was needed
                dataset['projection_params'] = None
            
            # Calculate diagonal for base size
            min_coords = np.min(boundary_points, axis=0)
            max_coords = np.max(boundary_points, axis=0)
            diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
            base_size = diagonal / (base_size_factor * 2.0)

            # Create boundary segments indices
            boundary_segments_indices = np.array([[i, (i + 1) % len(boundary_points)] for i in range(len(boundary_points))])

            # Combine boundary points (we don't need grid points if using hull segments)
            all_points = boundary_points
            
            # Use DirectTriangleWrapper
            from meshit.triangle_direct import DirectTriangleWrapper
            triangulator = DirectTriangleWrapper(
                gradient=gradient,
                min_angle=min_angle,
                base_size=base_size
            )
            
            # Set Triangle options
            triangle_options = f"pzq{min_angle}a{base_size*base_size*0.5}"
            triangulator.set_triangle_options(triangle_options)
            
            # Run triangulation
            triangulation_result = triangulator.triangulate(
                points=all_points,
                segments=boundary_segments_indices,
                uniform=True # Force uniform for consistency
            )
            
            # Get vertices and triangles
            vertices = triangulation_result['vertices']
            triangles = triangulation_result['triangles']
            
            # Project vertices back to 3D if needed
            if dataset['projection_params'] is not None:
                projection_params = dataset['projection_params']
                centroid = projection_params['centroid']
                basis = projection_params['basis']
                
                # Get original points with Z values
                original_points = dataset['points']
                original_boundary = dataset['hull_points'][:-1]  # Exclude closing point
                
                # Convert 2D triangulation vertices back to 3D
                vertices_3d = np.zeros((len(vertices), 3))
                
                for i, vertex_2d in enumerate(vertices):
                    # First check if this is exactly one of the boundary points
                    is_boundary_point = False
                    for j, bp_2d in enumerate(boundary_points):
                        if np.allclose(vertex_2d, bp_2d, atol=1e-10):
                            # Use original boundary point Z-value
                            vertices_3d[i] = original_boundary[j]
                            is_boundary_point = True
                            break
                    
                    if not is_boundary_point:
                        # Project back to 3D using the projection basis
                        vertex_3d = centroid.copy()  # Start with centroid
                        
                        # Add contributions from each basis vector
                        for j in range(2):  # 2D coordinates
                            vertex_3d += vertex_2d[j] * basis[j]  # basis[j] is a 3D vector
                        
                        # For interior points, find the closest original point 
                        # and use its Z value to better preserve 3D structure
                        vertex_3d_xy = vertex_3d[:2]  # Just X,Y for distance computation
                        
                        # Get distances to all original points (using just X,Y)
                        distances = np.sum((original_points[:, :2] - vertex_3d_xy)**2, axis=1)
                        
                        # Find nearest original point
                        closest_idx = np.argmin(distances)
                        
                        # Use Z from closest original point but X,Y from projection
                        vertex_3d[2] = original_points[closest_idx, 2]
                        
                        vertices_3d[i] = vertex_3d
                
                # Replace vertices with 3D version
                vertices = vertices_3d

            # Store results
            dataset['triangulation_result'] = {
                'vertices': vertices,
                'triangles': triangles,
                'uniform': uniform,
                'gradient': gradient,
                'min_angle': min_angle,
                'base_size': base_size,
            }

            return True # Indicate success

        except Exception as e:
            logger.error(f"Error triangulating {dataset_name}: {str(e)}")
            return False # Indicate error

    def run_triangulation(self):
        """Run triangulation on the segments and points of the *active* dataset (primarily for context menu)"""
        # Check if we have an active dataset
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.datasets):
            self.statusBar().showMessage("Error: No active dataset selected")
            QMessageBox.critical(self, "Error", "No active dataset selected")
            return

        self.statusBar().showMessage(f"Running triangulation for {self.datasets[self.current_dataset_index]['name']}...") # Add status message here for single run
        success = self._run_triangulation_for_dataset(self.current_dataset_index)

        if success:
            # Update statistics and visualization after computing for the active one
            self._update_statistics()
            self._visualize_all_triangulations()
            self._update_visualization() # Ensure other dependent views are updated/cleared
            self.notebook.setCurrentIndex(3)  # Switch to triangulation tab
            self.statusBar().showMessage(f"Completed triangulation for {self.datasets[self.current_dataset_index]['name']}") # Add success message here

    def run_all_triangulations(self):
        """Run triangulation for all datasets that have segments using a worker thread."""
        datasets_with_segments_indices = [i for i, d in enumerate(self.datasets) if d.get('segments') is not None]
        if not datasets_with_segments_indices:
            QMessageBox.information(self, "No Segments", "No datasets have computed segments. Please compute segments first.")
            return
        self._run_batch_computation("triangulations", len(datasets_with_segments_indices))
    
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
        """Export the triangulation results to a file for the active dataset"""
        self.statusBar().showMessage("Exporting results...")
        
        # Check if we have an active dataset
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.datasets):
            self.statusBar().showMessage("Error: No active dataset selected")
            QMessageBox.critical(self, "Error", "No active dataset selected")
            return
        
        # Get active dataset
        dataset = self.datasets[self.current_dataset_index]
        
        if dataset.get('triangulation_result') is None:
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
                vertices = dataset['triangulation_result']['vertices']
                triangles = dataset['triangulation_result']['triangles']
                
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
                # Check if vertex has Z coordinate
                if vertex.shape[0] >= 3:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                else:
                    # If only 2D, use 0 for Z
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
                # Check if vertex has Z coordinate
                if vertex.shape[0] >= 3:
                    f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
                else:
                    # If only 2D, use 0 for Z
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
                
                # Get Z coordinates or default to 0
                z1 = v1[2] if v1.shape[0] >= 3 else 0.0
                z2 = v2[2] if v2.shape[0] >= 3 else 0.0
                z3 = v3[2] if v3.shape[0] >= 3 else 0.0
                
                # Calculate normal (cross product of two sides)
                # For a non-flat mesh, calculate actual normal
                if v1.shape[0] >= 3 and v2.shape[0] >= 3 and v3.shape[0] >= 3:
                    # Simple normal calculation
                    side1 = np.array([v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]])
                    side2 = np.array([v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]])
                    normal = np.cross(side1, side2)
                    # Normalize
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal = normal / norm
                    f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                else:
                    # Default normal for flat mesh
                    f.write(f"  facet normal 0.0 0.0 1.0\n")
                
                f.write(f"    outer loop\n")
                f.write(f"      vertex {v1[0]} {v1[1]} {z1}\n")
                f.write(f"      vertex {v2[0]} {v2[1]} {z2}\n")
                f.write(f"      vertex {v3[0]} {v3[1]} {z3}\n")
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

    def _show_dataset_context_menu(self, position):
        """Show context menu for dataset list items"""
        if self.dataset_list_widget.count() == 0:
            return
            
        # Get selected item
        selected_items = self.dataset_list_widget.selectedItems()
        if not selected_items:
            return
            
        # Create context menu
        menu = QMenu()
        rename_action = menu.addAction("Rename")
        toggle_action = menu.addAction("Toggle Visibility")
        change_color_action = menu.addAction("Change Color")
        menu.addSeparator()
        
        # Add processing actions
        process_menu = menu.addMenu("Process")
        compute_hull_action = process_menu.addAction("Compute Hull")
        compute_segments_action = process_menu.addAction("Compute Segments")
        compute_triangulation_action = process_menu.addAction("Compute Triangulation")
        process_all_action = process_menu.addAction("Process All Steps")
        
        menu.addSeparator()
        remove_action = menu.addAction("Remove")
        
        # Show menu at position
        action = menu.exec_(self.dataset_list_widget.mapToGlobal(position))
        
        # Handle action
        if not action:
            return
            
        # Get selected dataset index
        selected_index = self.dataset_list_widget.row(selected_items[0])
        
        if action == rename_action:
            self._rename_dataset(selected_index)
        elif action == toggle_action:
            self._toggle_dataset_visibility(selected_index)
        elif action == change_color_action:
            self._change_dataset_color(selected_index)
        elif action == remove_action:
            self._remove_dataset(selected_index)
        elif action == compute_hull_action:
            self.current_dataset_index = selected_index
            self.compute_hull()
        elif action == compute_segments_action:
            self.current_dataset_index = selected_index
            self.compute_segments()
        elif action == compute_triangulation_action:
            self.current_dataset_index = selected_index
            self.run_triangulation()
        elif action == process_all_action:
            self.current_dataset_index = selected_index
            self._process_all_steps(selected_index)
    
    def _process_all_steps(self, dataset_index):
        """Process all steps (hull, segments, triangulation) for a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        # Set as active dataset
        self.current_dataset_index = dataset_index
        
        # Display processing message
        dataset_name = self.datasets[dataset_index]['name']
        self.statusBar().showMessage(f"Processing all steps for {dataset_name}...")
        
        try:
            # Step 1: Compute hull
            self.compute_hull()
            
            # Step 2: Compute segments
            self.compute_segments()
            
            # Step 3: Run triangulation
            self.run_triangulation()
            
            self.statusBar().showMessage(f"Completed all processing steps for {dataset_name}")
        except Exception as e:
            self.statusBar().showMessage(f"Error processing dataset: {str(e)}")
            logger.error(f"Error during dataset processing: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error processing dataset: {str(e)}")
    
    def process_all_datasets(self):
        """Process all steps for all datasets"""
        if not self.datasets:
            QMessageBox.information(self, "No Datasets", "No datasets available to process.")
            return
            
        num_processed = 0
        total_datasets = len(self.datasets)
        
        self.statusBar().showMessage(f"Processing all steps for {total_datasets} datasets...")
        
        try:
            # Process each dataset
            for i in range(total_datasets):
                dataset_name = self.datasets[i]['name']
                self.statusBar().showMessage(f"Processing {dataset_name} ({i+1}/{total_datasets})...")
                
                # Set as active
                self.current_dataset_index = i
                
                # Compute hull if needed
                if self.datasets[i].get('hull_points') is None:
                    # Call compute hull but catch any exceptions to continue with other datasets
                    try:
                        self.compute_hull()
                    except Exception as e:
                        logger.error(f"Error computing hull for {dataset_name}: {str(e)}")
                        continue
                
                # Compute segments if needed
                if self.datasets[i].get('segments') is None:
                    try:
                        self.compute_segments()
                    except Exception as e:
                        logger.error(f"Error computing segments for {dataset_name}: {str(e)}")
                        continue
                
                # Run triangulation if needed
                if self.datasets[i].get('triangulation_result') is None:
                    try:
                        self.run_triangulation()
                    except Exception as e:
                        logger.error(f"Error triangulating {dataset_name}: {str(e)}")
                        continue
                
                num_processed += 1
            
            # Show completion message
            self.statusBar().showMessage(f"Successfully processed {num_processed} out of {total_datasets} datasets")
            
            # Update visualizations for all
            self._update_visualization()
            
        except Exception as e:
            self.statusBar().showMessage(f"Error in batch processing: {str(e)}")
            logger.error(f"Error during batch processing: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error in batch processing: {str(e)}")
    
    def _on_dataset_selection_changed(self):
        """Handle dataset selection change"""
        selected_items = self.dataset_list_widget.selectedItems()
        if not selected_items:
            return
            
        # Get selected dataset index
        selected_index = self.dataset_list_widget.row(selected_items[0])
        
        # Set as active dataset
        self.current_dataset_index = selected_index
        
        # Update statistics and visualization for the selected dataset
        self._update_statistics()
        self._update_visualization()
        
    def _update_statistics(self):
        """Update statistics labels based on loaded datasets"""
        num_datasets = len(self.datasets)
        self.num_datasets_label.setText(f"Datasets: {num_datasets}")
        
        if num_datasets == 0:
            self.points_label.setText("Points: 0")
            self.bounds_label.setText("Bounds: N/A")
            self.hull_points_label.setText("Hull vertices: 0")
            self.hull_area_label.setText("Hull area: 0.0")
            self.num_segments_label.setText("Segments: 0")
            self.avg_segment_length_label.setText("Avg length: 0.0")
            self.num_triangles_label.setText("Triangles: 0")
            self.num_vertices_label.setText("Vertices: 0")
            self.mean_edge_label.setText("Mean edge: 0.0")
            self.uniformity_label.setText("Uniformity: 0.0")
            return
        
        # Update statistics for the active dataset
        if 0 <= self.current_dataset_index < num_datasets:
            dataset = self.datasets[self.current_dataset_index]
            
            # Points statistics
            if dataset['points'] is not None:
                self.points_label.setText(f"Points: {len(dataset['points'])}")
                
                # Calculate bounds - only use X and Y for display
                min_coords = np.min(dataset['points'], axis=0)
                max_coords = np.max(dataset['points'], axis=0)
                min_x, min_y = min_coords[0], min_coords[1]
                max_x, max_y = max_coords[0], max_coords[1]
                
                # If 3D data is available, also show Z range
                if dataset['points'].shape[1] >= 3:
                    min_z, max_z = min_coords[2], max_coords[2]
                    self.bounds_label.setText(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}] Z[{min_z:.2f},{max_z:.2f}]")
                else:
                    self.bounds_label.setText(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}]")
            else:
                self.points_label.setText("Points: 0")
                self.bounds_label.setText("Bounds: N/A")
            
            # Hull statistics
            if dataset.get('hull_points') is not None:
                hull_points = dataset['hull_points']
                self.hull_points_label.setText(f"Hull vertices: {len(hull_points)-1}")  # -1 for the closing point
                
                # Calculate hull area (approximate)
                area = 0.0
                for i in range(len(hull_points)-1):
                    # Extract x,y coordinates only
                    x1, y1 = hull_points[i][0], hull_points[i][1]
                    x2, y2 = hull_points[i+1][0], hull_points[i+1][1]
                    area += x1*y2 - x2*y1
                area = abs(area) / 2.0
                self.hull_area_label.setText(f"Hull area: {area:.2f}")
            else:
                self.hull_points_label.setText("Hull vertices: 0")
                self.hull_area_label.setText("Hull area: 0.0")
            
            # Segment statistics
            if dataset.get('segments') is not None:
                segments = dataset['segments']
                self.num_segments_label.setText(f"Segments: {len(segments)}")
                
                # Calculate average segment length
                segment_lengths = [np.linalg.norm(segment[1] - segment[0]) for segment in segments]
                avg_length = np.mean(segment_lengths)
                self.avg_segment_length_label.setText(f"Avg length: {avg_length:.2f}")
            else:
                self.num_segments_label.setText("Segments: 0")
                self.avg_segment_length_label.setText("Avg length: 0.0")
            
            # Triangulation statistics
            if dataset.get('triangulation_result') is not None:
                triangulation = dataset['triangulation_result']
                vertices = triangulation['vertices']
                triangles = triangulation['triangles']
                
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
                
                if edge_lengths:
                    mean_edge = np.mean(edge_lengths)
                    edge_std = np.std(edge_lengths)
                    uniformity = edge_std / mean_edge if mean_edge > 0 else 0
                    
                    self.mean_edge_label.setText(f"Mean edge: {mean_edge:.4f}")
                    self.uniformity_label.setText(f"Uniformity: {uniformity:.4f}")
                else:
                    self.mean_edge_label.setText("Mean edge: 0.0")
                    self.uniformity_label.setText("Uniformity: 0.0")
            else:
                self.num_triangles_label.setText("Triangles: 0")
                self.num_vertices_label.setText("Vertices: 0")
                self.mean_edge_label.setText("Mean edge: 0.0")
                self.uniformity_label.setText("Uniformity: 0.0")
    
    def _update_dataset_list(self):
        """Update the dataset list widget"""
        # Save current selection
        current_index = self.current_dataset_index
        
        # Clear list
        self.dataset_list_widget.clear()
        
        # Add datasets to list
        for dataset in self.datasets:
            visibility = "✓" if dataset.get('visible', True) else "✗"
            color_square = "■ "
            item_text = f"{color_square}{visibility} {dataset['name']}"
            
            item = QListWidgetItem(item_text)
            
            # Set item color based on dataset color
            color = dataset.get('color', '#000000')
            item.setForeground(QColor(color))
            
            self.dataset_list_widget.addItem(item)
        
        # Restore selection if possible
        if 0 <= current_index < self.dataset_list_widget.count():
            self.dataset_list_widget.setCurrentRow(current_index)
        elif self.dataset_list_widget.count() > 0:
            self.dataset_list_widget.setCurrentRow(0)
            self.current_dataset_index = 0
        else:
            self.current_dataset_index = -1
    
    def _rename_dataset(self, dataset_index):
        """Rename a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        dataset = self.datasets[dataset_index]
        old_name = dataset['name']
        
        # Create dialog for renaming
        dialog = QDialog(self)
        dialog.setWindowTitle("Rename Dataset")
        dialog.setFixedSize(300, 100)
        dialog_layout = QVBoxLayout(dialog)
        
        # Add name input
        dialog_layout.addWidget(QLabel("Dataset Name:"))
        name_input = QLineEdit(old_name)
        dialog_layout.addWidget(name_input)
        
        # Add buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        ok_btn = QPushButton("OK")
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        dialog_layout.addLayout(button_layout)
        
        # Connect signals
        cancel_btn.clicked.connect(dialog.reject)
        ok_btn.clicked.connect(dialog.accept)
        
        # Show dialog and get result
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            new_name = name_input.text().strip()
            if new_name:
                # Update dataset name
                dataset['name'] = new_name
                
                # Update list
                self._update_dataset_list()
    
    def _toggle_dataset_visibility(self, dataset_index):
        """Toggle visibility of a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        dataset = self.datasets[dataset_index]
        dataset['visible'] = not dataset.get('visible', True)
        
        # Update list
        self._update_dataset_list()
        
        # Update visualization
        self._update_visualization()
    
    def _change_dataset_color(self, dataset_index):
        """Change color of a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        dataset = self.datasets[dataset_index]
        old_color = dataset.get('color', '#000000')
        
        # Get new color using color dialog
        new_color = QColorDialog.getColor(QColor(old_color), self, "Select Dataset Color")
        
        if new_color.isValid():
            # Update dataset color
            dataset['color'] = new_color.name()
            
            # Update list
            self._update_dataset_list()
            
            # Update visualization
            self._update_visualization()
    
    def _remove_dataset(self, dataset_index):
        """Remove a dataset"""
        if not (0 <= dataset_index < len(self.datasets)):
            return
            
        # Confirm deletion
        confirm = QMessageBox.question(
            self, 
            "Remove Dataset", 
            f"Are you sure you want to remove dataset '{self.datasets[dataset_index]['name']}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Remove dataset
            self.datasets.pop(dataset_index)
            
            # Update dataset list
            self._update_dataset_list()
            
            # Update statistics and visualization
            self._update_statistics()
            self._update_visualization()
    
    def clear_all_datasets(self):
        """Clear all loaded datasets"""
        if not self.datasets:
            return
            
        # Confirm deletion
        confirm = QMessageBox.question(
            self, 
            "Clear All Datasets", 
            f"Are you sure you want to remove all {len(self.datasets)} datasets?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            # Clear datasets
            self.datasets = []
            self.current_dataset_index = -1
            
            # Update UI
            self._update_dataset_list()
            self._update_statistics()
            
            # Clear visualizations
            self._clear_visualizations()
            
            self.statusBar().showMessage("All datasets cleared")
    
    def remove_active_dataset(self):
        """Remove the active dataset"""
        if 0 <= self.current_dataset_index < len(self.datasets):
            self._remove_dataset(self.current_dataset_index)
    
    def _clear_visualizations(self):
        """Clear all visualizations"""
        self._clear_hull_plot()
        self._clear_segment_plot()
        self._clear_triangulation_plot()
        
        # Clear points visualization
        while self.file_viz_layout.count():
            item = self.file_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Load data to visualize points in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nLoad data to visualize points in 2D."
            
        self.file_viz_placeholder = QLabel(placeholder_text)
        self.file_viz_placeholder.setAlignment(Qt.AlignCenter)
        self.file_viz_layout.addWidget(self.file_viz_placeholder)
    
    def _update_visualization(self):
        """Update all visualizations based on loaded datasets"""
        # Check if we have any datasets
        if not self.datasets:
            self._clear_visualizations()
            return
        
        # Get visible datasets
        visible_datasets = [d for d in self.datasets if d.get('visible', True)]
        
        if not visible_datasets:
            self._clear_visualizations()
            return
        
        # Update points visualization
        self._visualize_all_points()
        
        # Update hull visualization if we have any hulls
        hulls_exist = any(d.get('hull_points') is not None for d in visible_datasets)
        if hulls_exist:
            self._visualize_all_hulls()
        
        # Update segments visualization if we have any segments
        segments_exist = any(d.get('segments') is not None for d in visible_datasets)
        if segments_exist:
            self._visualize_all_segments()
        
        # Update triangulation visualization if we have any triangulations
        triangulations_exist = any(d.get('triangulation_result') is not None for d in visible_datasets)
        if triangulations_exist:
            self._visualize_all_triangulations()
    
    def load_file(self):
        """Load a single data file"""
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
                # Create a new dataset
                filename = os.path.basename(file_path)
                dataset = {
                    'name': filename,
                    'points': points,
                    'visible': True,
                    'color': self._get_next_color()
                }
                
                # Add to datasets
                self.datasets.append(dataset)
                self.current_dataset_index = len(self.datasets) - 1
                
                # Update UI
                self._update_dataset_list()
                self._update_statistics()
                self._visualize_all_points()
                
                self.statusBar().showMessage(f"Successfully loaded {len(points)} points from {filename}")
            else:
                self.statusBar().showMessage("Error: No valid points found in file")
                QMessageBox.critical(self, "Error", "No valid points found in file")
        
        except Exception as e:
            self.statusBar().showMessage(f"Error loading file: {str(e)}")
            logger.error(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
    
    def load_multiple_files(self):
        """Load multiple data files"""
        self.statusBar().showMessage("Loading multiple files...")
        
        # Open file dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select point data files",
            "",
            "Text files (*.txt);;Data files (*.dat);;CSV files (*.csv);;All files (*.*)"
        )
        
        if not file_paths:
            self.statusBar().showMessage("File loading canceled")
            return
        
        successful_loads = 0
        
        for file_path in file_paths:
            try:
                # Update status with file path
                filename = os.path.basename(file_path)
                self.statusBar().showMessage(f"Loading file: {filename}...")
                
                # Try to read the file
                points = self._read_point_file(file_path)
                
                if points is not None and len(points) > 0:
                    # Create a new dataset
                    dataset = {
                        'name': filename,
                        'points': points,
                        'visible': True,
                        'color': self._get_next_color()
                    }
                    
                    # Add to datasets
                    self.datasets.append(dataset)
                    self.current_dataset_index = len(self.datasets) - 1
                    successful_loads += 1
                else:
                    logger.warning(f"No valid points found in file: {filename}")
            except Exception as e:
                logger.error(f"Error loading file {filename}: {str(e)}")
        
        if successful_loads > 0:
            # Update UI
            self._update_dataset_list()
            self._update_statistics()
            self._visualize_all_points()
            
            self.statusBar().showMessage(f"Successfully loaded {successful_loads} out of {len(file_paths)} files")
        else:
            self.statusBar().showMessage("Error: No valid points found in any file")
            QMessageBox.critical(self, "Error", "No valid points found in any file")
            
    def _get_next_color(self):
        """Get next color from palette for a new dataset"""
        if not self.datasets:
            return self.DEFAULT_COLORS[0]
        
        # Count existing colors
        used_colors = {}
        for dataset in self.datasets:
            color = dataset.get('color')
            if color:
                used_colors[color] = used_colors.get(color, 0) + 1
        
        # Find first least used color
        for color in self.DEFAULT_COLORS:
            if color not in used_colors:
                return color
        
        # If all colors are used, use the least used one
        min_used = min(used_colors.values())
        least_used_colors = [c for c, count in used_colors.items() if count == min_used]
        
        # Return the first one from palette order
        for color in self.DEFAULT_COLORS:
            if color in least_used_colors:
                return color
        
        # If all else fails, return a random color
        return self.DEFAULT_COLORS[np.random.randint(0, len(self.DEFAULT_COLORS))]

    def _visualize_all_points(self):
        """Visualize all visible datasets' points"""
        # Get visible datasets
        visible_datasets = [d for d in self.datasets if d.get('visible', True)]
        
        if not visible_datasets:
            self._clear_hull_plot()
            return
        
        # Clear existing visualization
        while self.file_viz_layout.count():
            item = self.file_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Calculate global bounds for all visible datasets
        all_points = np.vstack([d['points'] for d in visible_datasets if d['points'] is not None])
        if len(all_points) == 0:
            return
            
        # Calculate bounds, correctly handling 3D coordinates
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        min_x, min_y = min_coords[0], min_coords[1]
        max_x, max_y = max_coords[0], max_coords[1]
        
        # Use 3D visualization if enabled
        if self.view_3d_enabled:
            # Create 3D visualization
            self._create_multi_dataset_3d_visualization(
                self.file_viz_frame,
                visible_datasets,
                "Points Visualization",
                view_type="points"
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot each dataset with its color
            for dataset in visible_datasets:
                points = dataset['points']
                if points is not None and len(points) > 0:
                    color = dataset.get('color', 'blue')
                    name = dataset.get('name', 'Unnamed')
                    ax.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.7, label=name)
            
            ax.set_aspect('equal')
            ax.set_title("Point Clouds")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            padding = max((max_x - min_x), (max_y - min_y)) * 0.05
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.file_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.file_viz_frame)
            self.file_viz_layout.addWidget(toolbar)
    
    def _visualize_all_hulls(self):
        """Visualize all visible hulls"""
        # Get datasets with hulls
        datasets_with_hull = []
        for dataset in self.datasets:
            if dataset.get('visible', True) and dataset.get('hull_points') is not None:
                datasets_with_hull.append(dataset)
                
        # Clear previous visualization if no datasets with hull
        if not datasets_with_hull:
            self._clear_hull_plot()
            return
            
        # Use 3D visualization or PyVista
        if self.view_3d_enabled:
            self._create_multi_dataset_3d_visualization(
                self.hull_viz_frame, 
                datasets_with_hull,
                "Convex Hull Visualization",
                view_type="hulls"
            )
            return
        
        # Clear previous visualization
        while self.hull_viz_layout.count():
            item = self.hull_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        # Create a matplotlib visualization
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Find global limits for all datasets
        all_points = []
        for dataset in datasets_with_hull:
            points = dataset.get('points')
            if points is not None and len(points) > 0:
                all_points.append(points)
                
        if not all_points:
            return
            
        all_points_np = np.vstack(all_points)
        min_x, min_y = np.min(all_points_np[:, 0:2], axis=0)
        max_x, max_y = np.max(all_points_np[:, 0:2], axis=0)
        
        # Add some margin to the plot
        margin_x = 0.1 * (max_x - min_x)
        margin_y = 0.1 * (max_y - min_y)
        
        # Plot each dataset
        for dataset in datasets_with_hull:
            points = dataset.get('points')
            hull_points = dataset.get('hull_points')
            color = dataset.get('color', '#000000')
            name = dataset.get('name', 'Unnamed')
            
            if points is None or len(points) == 0 or hull_points is None or len(hull_points) == 0:
                continue
            
            # Plot points (with reduced alpha for better visualization)
            ax.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.3)
            
            # Plot hull (using first 2 dimensions)
            ax.plot(hull_points[:, 0], hull_points[:, 1], c=color, linewidth=2, label=name)
            
        ax.set_aspect('equal')
        ax.set_title(f"Hull Visualization: {len(datasets_with_hull)} datasets")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
        
        # Set limits with margin
        ax.set_xlim(min_x - margin_x, max_x + margin_x)
        ax.set_ylim(min_y - margin_y, max_y + margin_y)
        
        # Create canvas
        canvas = FigureCanvas(fig)
        self.hull_viz_layout.addWidget(canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar(canvas, self.hull_viz_frame)
        self.hull_viz_layout.addWidget(toolbar)
    
    def _visualize_all_segments(self):
        """Visualize all visible datasets' segments"""
        # Get visible datasets with segments
        visible_datasets = [d for d in self.datasets if d.get('visible', True) and d.get('segments') is not None]
        
        if not visible_datasets:
            self._clear_segment_plot()
            return
        
        # Clear existing visualization
        while self.segment_viz_layout.count():
            item = self.segment_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Calculate global bounds for all visible datasets
        all_points = np.vstack([d['points'] for d in visible_datasets if d['points'] is not None])
        if len(all_points) == 0:
            return
            
        # Handle 3D coordinates properly
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        min_x, min_y = min_coords[0], min_coords[1]
        max_x, max_y = max_coords[0], max_coords[1]
        
        # Use 3D visualization if enabled
        if self.view_3d_enabled:
            # Create 3D visualization - pass segments to be drawn as lines
            self._create_multi_dataset_3d_visualization(
                self.segment_viz_frame,
                visible_datasets,
                "Segmentation Visualization",
                view_type="segments" # This already handles plotting segments
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot each dataset with its color
            for dataset in visible_datasets:
                points = dataset['points']
                # hull_points = dataset.get('hull_points') # No longer needed for plotting here
                segments = dataset['segments']
                
                if points is not None and len(points) > 0 and segments is not None and len(segments) > 0:
                    color = dataset.get('color', 'blue')
                    name = dataset.get('name', 'Unnamed')
                    
                    # Plot points
                    ax.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=0.3)
                    
                    # # Plot hull if available - REMOVED to show segment boundary instead
                    # if hull_points is not None and len(hull_points) > 0:
                    #     ax.plot(hull_points[:, 0], hull_points[:, 1],
                    #           color=color, linewidth=1, alpha=0.5)

                    # Plot segments - make them slightly more prominent
                    for segment in segments:
                        ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 
                              color=color, linewidth=1.8, alpha=0.9) # Increased linewidth/alpha
                    
                    # Plot segment endpoints
                    all_endpoints = np.vstack([segment[0] for segment in segments] + 
                                            [segment[1] for segment in segments])
                    unique_endpoints = np.unique(all_endpoints, axis=0)
                    ax.scatter(unique_endpoints[:, 0], unique_endpoints[:, 1], 
                             s=20, c=color, edgecolor='black', label=f"{name} Boundary Points") # Adjusted label
            
            ax.set_aspect('equal')
            ax.set_title("Segmentations (Boundary Defined by Segments)") # Updated title
            # ... rest of matplotlib setup (labels, legend, grid, limits) ...
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            padding = max((max_x - min_x), (max_y - min_y)) * 0.05
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.segment_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.segment_viz_frame)
            self.segment_viz_layout.addWidget(toolbar)
        
        # Update the segments legend (remains the same)
        # ... existing legend update code ...
    
    def _visualize_all_triangulations(self):
        """Visualize all visible datasets' triangulations"""
        # Get visible datasets with triangulations
        visible_datasets = [d for d in self.datasets if d.get('visible', True) and d.get('triangulation_result') is not None]
        
        if not visible_datasets:
            self._clear_triangulation_plot()
            return
        
        # Clear existing visualization
        while self.tri_viz_layout.count():
            item = self.tri_viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Calculate global bounds for all visible datasets
        all_vertices = []
        for dataset in visible_datasets:
            if dataset.get('triangulation_result') is not None:
                vertices = dataset['triangulation_result']['vertices']
                if vertices is not None and len(vertices) > 0:
                    all_vertices.append(vertices)
        
        if not all_vertices:
            return
            
        all_vertices = np.vstack(all_vertices)
        # Handle 3D coordinates properly
        min_coords = np.min(all_vertices, axis=0)
        max_coords = np.max(all_vertices, axis=0)
        min_x, min_y = min_coords[0], min_coords[1]
        max_x, max_y = max_coords[0], max_coords[1]
        
        # Use 3D visualization if enabled
        if self.view_3d_enabled:
            # In 3D, the mesh itself shows the boundary. We can optionally highlight edges.
            # The existing _create_multi_dataset_3d_visualization already shows edges.
            self._create_multi_dataset_3d_visualization(
                self.tri_viz_frame,
                visible_datasets,
                "Triangulation Visualization",
                view_type="triangulation"
            )
        else:
            # Fall back to matplotlib if PyVista is not available
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot each dataset with its color
            for dataset in visible_datasets:
                triangulation_result = dataset['triangulation_result']
                # hull_points = dataset.get('hull_points') # Don't use original hull
                
                if triangulation_result is not None:
                    vertices = triangulation_result.get('vertices')
                    triangles = triangulation_result.get('triangles')
                    
                    if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                        color = dataset.get('color', 'blue')
                        name = dataset.get('name', 'Unnamed')
                        
                        # Plot triangulation using triplot
                        from matplotlib.tri import Triangulation
                        tri = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
                        ax.triplot(tri, color=color, lw=0.5, alpha=0.7, label=f"{name} Mesh")
                        
                        # --- Plot actual boundary edges ---
                        boundary_edges_indices = self._get_boundary_edges(triangles)
                        if boundary_edges_indices:
                            boundary_lines = []
                            for i1, i2 in boundary_edges_indices:
                                p1 = vertices[i1]
                                p2 = vertices[i2]
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.5, label=f"{name} Boundary" if not ax.get_legend_handles_labels()[1] else "") # Only label once per dataset
                        # --- End plot boundary edges ---

                        # # Plot hull boundary if available - REMOVED
                        # if hull_points is not None and len(hull_points) > 3:
                        #     ax.plot(hull_points[:, 0], hull_points[:, 1], color=color, linewidth=1.5)
            
            ax.set_aspect('equal')
            ax.set_title("Triangulations (Actual Boundary)") # Updated title
            # ... rest of matplotlib setup (labels, legend, grid, limits) ...
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
             # Consolidate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles)) # Remove duplicate labels
            ax.legend(by_label.values(), by_label.keys())
            
            # Add grid and set limits with some padding
            ax.grid(True, linestyle='--', alpha=0.6)
            padding = max((max_x - min_x), (max_y - min_y)) * 0.05
            ax.set_xlim(min_x - padding, max_x + padding)
            ax.set_ylim(min_y - padding, max_y + padding)
            
            # Create canvas
            canvas = FigureCanvas(fig)
            self.tri_viz_layout.addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.tri_viz_frame)
            self.tri_viz_layout.addWidget(toolbar)
        
        # Update the triangulation legend (remains the same)
        # ... existing legend update code ...
    
    def _create_multi_dataset_3d_visualization(self, parent_frame, datasets, title, view_type="points"):
        """Create a 3D visualization of multiple datasets
        
        Args:
            parent_frame: Frame to contain the visualization
            datasets: List of datasets to visualize
            title: Title for the visualization
            view_type: Type of visualization: "points", "hulls", "segments", or "triangulation"
        """
        if not HAVE_PYVISTA:
            # Clear previous content if any
            while parent_frame.layout() and parent_frame.layout().count():
                item = parent_frame.layout().takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            # Ensure parent_frame has a layout
            if parent_frame.layout() is None:
                parent_frame.setLayout(QVBoxLayout())

            # Create a message if PyVista is not available
            msg_widget = QWidget()
            msg_layout = QVBoxLayout(msg_widget)
            msg = QLabel("PyVista not installed.\nPlease install PyVista for 3D visualization.")
            msg.setAlignment(Qt.AlignCenter)
            msg_layout.addWidget(msg)
            parent_frame.layout().addWidget(msg_widget)
            return msg_widget # Return the message widget
        
        # Close previous plotter if it exists
        if hasattr(self, 'current_plotter') and self.current_plotter is not None:
            try:
                # Explicitly close and maybe delete the plotter resources
                self.current_plotter.close()
            except Exception as e:
                logger.warning(f"Error closing previous plotter: {e}")
            self.current_plotter = None # Ensure reference is cleared

        # Clear previous content from parent frame layout first
        parent_layout = parent_frame.layout()
        if parent_layout is None:
            # If no layout exists, create one
            parent_layout = QVBoxLayout(parent_frame)
            parent_layout.setContentsMargins(0,0,0,0) # Ensure no margins interfer

        while parent_layout.count():
            item = parent_layout.takeAt(0)
            widget = item.widget()
            if widget:
                # Critical: Ensure the plotter widget is properly deleted
                if isinstance(widget, QFrame) and hasattr(widget, 'interactor'): # Check if it's a plotter frame
                    try:
                        widget.interactor.close() # Try closing interactor too
                    except Exception as e:
                        logger.debug(f"Minor issue closing interactor: {e}")
                widget.deleteLater()


        # Create a container widget for the entire visualization area
        vis_container_widget = QWidget()
        vis_container_layout = QVBoxLayout(vis_container_widget)
        vis_container_layout.setContentsMargins(0, 0, 0, 0) # Remove margins

        # Add visualization info header based on view_type
        if view_type == "points":
            info_title = "Point Cloud Visualization"
        elif view_type == "hulls":
            info_title = "Convex Hull Visualization"
        elif view_type == "segments":
            info_title = "Segmentation Visualization"
        elif view_type == "triangulation":
            info_title = "Triangulation Visualization"
        else:
            info_title = title # Fallback to provided title
            
        info_label = QLabel(info_title)
        info_label.setAlignment(Qt.AlignCenter)
        vis_container_layout.addWidget(info_label) # Add title to the main container
        
        # Create legend with colored boxes for each dataset
        legend_widget = QWidget() # Widget to hold the legend layout
        legend_layout = QHBoxLayout(legend_widget)
        legend_layout.setContentsMargins(5, 2, 5, 2) # Small margins for legend

        visible_datasets_in_list = [d for d in datasets if d.get('visible', True)] # Filter visible for legend
        for dataset in visible_datasets_in_list:
            color = dataset.get('color', '#000000')
            name = dataset.get('name', 'Unnamed')
            
            legend_item = QWidget()
            legend_item_layout = QHBoxLayout(legend_item)
            legend_item_layout.setContentsMargins(0, 0, 5, 0) # Spacing between legend items
            
            # Color box
            color_box = QLabel("■")
            color_box.setStyleSheet(f"color: {color}; font-size: 16px;")
            legend_item_layout.addWidget(color_box)
            
            # Dataset name
            name_label = QLabel(name)
            legend_item_layout.addWidget(name_label)
            
            legend_layout.addWidget(legend_item)
            
        # Add spacer to push items to the left
        legend_layout.addStretch()
        vis_container_layout.addWidget(legend_widget) # Add legend widget to the main container
        
        try:
            # Create PyVista plotter widget using QtInteractor
            from pyvistaqt import QtInteractor
            
            # *** Visualization Fix: Create plotter without intermediate frame ***
            # QtInteractor itself is a QFrame, so we can add it directly
            # parent=vis_container_widget makes it live inside the main container
            self.current_plotter = QtInteractor(parent=vis_container_widget)
            vis_container_layout.addWidget(self.current_plotter) # Add the plotter QFrame
            
            # Set background color
            self.current_plotter.set_background("#383F51")
            
            # Collect all points for global bounds
            all_points = []
            for dataset in datasets:
                if dataset.get('visible', True) and dataset.get('points') is not None:
                    all_points.append(dataset['points'])
            
            # Calculate dataset extents if we have points - for informational purposes only
            if all_points:
                all_points_np = np.vstack(all_points)
                if len(all_points_np) > 0:
                    logger.debug(f"Visualizing {len(all_points_np)} points across {len(all_points)} datasets")
                else:
                    logger.warning("No points found in visible datasets to calculate range.")
            else:
                logger.warning("No visible datasets with points found.")
            
            # Process datasets based on visualization type
            plotter_has_geometry = False # Flag to check if anything was added
            for i, dataset in enumerate(datasets):
                if not dataset.get('visible', True):
                    continue
                    
                # Get dataset properties
                points = dataset.get('points')
                color = dataset.get('color', '#000000')
                name = dataset.get('name', 'Unnamed')
                
                if points is None or len(points) == 0:
                    continue
                
                # Convert 2D points to 3D with height variation
                if points.shape[1] == 2:
                    points_3d = np.zeros((len(points), 3))
                    points_3d[:, 0] = points[:, 0]
                    points_3d[:, 1] = points[:, 1]
                    # Z coordinate is left as 0 for 2D points
                elif points.shape[1] >= 3:
                    # Use actual Z coordinates from the data
                    points_3d = points.copy()[:, 0:3]  # Use the first 3 dimensions only
                else:
                    logger.warning(f"Dataset '{name}' has unexpected point dimensions: {points.shape[1]}. Skipping.")
                    continue

                
                # Add visualization based on type
                if view_type == "points":
                    point_cloud = pv.PolyData(points_3d)
                    self.current_plotter.add_mesh(point_cloud, color=color, render_points_as_spheres=True, 
                                         point_size=8, label=name)
                    plotter_has_geometry = True
                
                elif view_type == "hulls":
                    hull_points = dataset.get('hull_points')
                    if hull_points is not None and len(hull_points) > 3:
                        # Use hull points directly - they should already have proper 3D coordinates
                        hull_3d = hull_points
                        
                        # Create lines for the hull
                        for j in range(len(hull_3d)-1):
                            line = pv.Line(hull_3d[j], hull_3d[j+1])
                            self.current_plotter.add_mesh(line, color=color, line_width=3)
                            
                        # Add the original points with reduced opacity for context
                        point_cloud = pv.PolyData(points_3d)
                        self.current_plotter.add_mesh(point_cloud, color=color, opacity=0.3, 
                                           render_points_as_spheres=True, point_size=5)
                        plotter_has_geometry = True
                    # Optionally add points too for context
                    point_cloud = pv.PolyData(points_3d)
                    self.current_plotter.add_mesh(point_cloud, color=color, opacity=0.3, render_points_as_spheres=True,
                                     point_size=5)
                

                elif view_type == "segments":
                    segments = dataset.get('segments')
                    if segments is not None and len(segments) > 0:
                        # Add points for context
                        point_cloud = pv.PolyData(points_3d)
                        self.current_plotter.add_mesh(point_cloud, color=color, opacity=0.3, render_points_as_spheres=True,
                                         point_size=5)

                        # Create 3D lines for segments - handle both list and numpy array formats
                        for segment in segments:
                            line_points = np.zeros((2, 3))
                            
                            # Handle both 2D and 3D segment points
                            if hasattr(segment[0], 'shape'):
                                # For numpy arrays
                                if segment[0].shape[0] == 2:
                                    # For 2D segments, copy X,Y and find Z from closest point
                                    line_points[0, 0:2] = segment[0]
                                    line_points[1, 0:2] = segment[1]
                                    
                                    # Set Z values based on closest original points
                                    for j in range(2):
                                        point_2d = segment[j]
                                        # Find closest point in original dataset for Z value
                                        distances = np.sum((points_3d[:, 0:2] - point_2d)**2, axis=1)
                                        closest_idx = np.argmin(distances)
                                        line_points[j, 2] = points_3d[closest_idx, 2]
                                else:
                                    # For 3D segments, use all dimensions directly
                                    line_points[0] = segment[0]
                                    line_points[1] = segment[1]
                            else:
                                # For list-based points
                                if len(segment[0]) == 2:
                                    # For 2D segments, copy X,Y and find Z from closest point
                                    line_points[0, 0] = segment[0][0]
                                    line_points[0, 1] = segment[0][1]
                                    line_points[1, 0] = segment[1][0]
                                    line_points[1, 1] = segment[1][1]
                                    
                                    # Set Z values based on closest original points
                                    for j in range(2):
                                        point_2d = np.array([segment[j][0], segment[j][1]])
                                        # Find closest point in original dataset for Z value
                                        distances = np.sum((points_3d[:, 0:2] - point_2d)**2, axis=1)
                                        closest_idx = np.argmin(distances)
                                        line_points[j, 2] = points_3d[closest_idx, 2]
                                else:
                                    # For 3D segments, use all dimensions directly
                                    line_points[0, 0] = segment[0][0]
                                    line_points[0, 1] = segment[0][1]
                                    line_points[0, 2] = segment[0][2]
                                    line_points[1, 0] = segment[1][0]
                                    line_points[1, 1] = segment[1][1]
                                    line_points[1, 2] = segment[1][2]
                            
                            line_obj = pv.Line(line_points[0], line_points[1])
                            self.current_plotter.add_mesh(line_obj, color=color, line_width=2.5)
                        plotter_has_geometry = True

                elif view_type == "triangulation":
                    triangulation_result = dataset.get('triangulation_result')
                    if triangulation_result is not None:
                        vertices = triangulation_result.get('vertices')
                        triangles = triangulation_result.get('triangles')
                        
                        if vertices is not None and len(vertices) > 0 and triangles is not None and len(triangles) > 0:
                            # Convert vertices to 3D, respecting Z coordinates 
                            if vertices.shape[1] == 2:
                                vertices_3d = np.zeros((len(vertices), 3))
                                vertices_3d[:, 0:2] = vertices
                                
                                # Try to map Z values from original points
                                for j in range(len(vertices)):
                                    vertex_2d = vertices[j]
                                    # Find closest point in original dataset for Z value
                                    distances = np.sum((points_3d[:, 0:2] - vertex_2d)**2, axis=1)
                                    closest_idx = np.argmin(distances)
                                    vertices_3d[j, 2] = points_3d[closest_idx, 2]
                            else:
                                # Use Z values if they're already present
                                vertices_3d = vertices.copy()[:, 0:3]
                            
                            cells = np.hstack([np.full((len(triangles), 1), 3), triangles])
                            mesh = pv.PolyData(vertices_3d, cells)
                            
                            self.current_plotter.add_mesh(mesh, color=color, opacity=0.7, 
                                                      show_edges=True, edge_color=color, 
                                                      line_width=1, specular=0.5, label=name)
                            plotter_has_geometry = True
            
            
            # Add axes and reset camera only if something was plotted
            if plotter_has_geometry:
                self.current_plotter.add_axes()
                self.current_plotter.reset_camera()
            else:
                logger.warning("No geometry added to the plotter for the current view.")
            
            # Add controls for adjustment
            controls_widget = QWidget() # Widget to hold controls
            controls_layout = QHBoxLayout(controls_widget)
            controls_layout.setContentsMargins(5, 2, 5, 2) # Small margins
            
            # Height adjustment slider - for Z axis exaggeration
            controls_layout.addWidget(QLabel("Z Exaggeration:"))
            height_slider = QSlider(Qt.Horizontal)
            height_slider.setMinimum(1)
            height_slider.setMaximum(100)
            # Reflect current height factor in slider position
            height_slider.setValue(int(self.height_factor * 20)) # Scale to slider range
            # Update height factor and re-trigger the visualization update for the current tab
            height_slider.valueChanged.connect(lambda v: self._set_height_factor_and_update(v / 20.0)) # Scale back from slider range
            controls_layout.addWidget(height_slider)
            
            # Add zoom controls
            controls_layout.addStretch(1) # Push zoom controls to the right
            controls_layout.addWidget(QLabel("Zoom:"))
            zoom_in_btn = QPushButton("+")
            zoom_in_btn.setMaximumWidth(30)
            zoom_in_btn.clicked.connect(lambda: self.current_plotter.camera.zoom(1.2))
            controls_layout.addWidget(zoom_in_btn)
            
            zoom_out_btn = QPushButton("-")
            zoom_out_btn.setMaximumWidth(30)
            zoom_out_btn.clicked.connect(lambda: self.current_plotter.camera.zoom(1/1.2))
            controls_layout.addWidget(zoom_out_btn)
            
            # Reset view button
            reset_btn = QPushButton("Reset View")
            reset_btn.clicked.connect(lambda: self.current_plotter.reset_camera())
            controls_layout.addWidget(reset_btn)
            
            vis_container_layout.addWidget(controls_widget) # Add controls below the plotter

        except ImportError:
            # This case should be caught by HAVE_PYVISTA check at the top
            logger.error("PyVistaQt import failed unexpectedly.")
            error_msg = QLabel("Error: Failed to load PyVistaQt.")
            error_msg.setAlignment(Qt.AlignCenter)
            vis_container_layout.addWidget(error_msg)
        except Exception as e:
            # Fallback if QtInteractor fails
            error_msg_text = f"Error creating 3D view: {str(e)}\nCheck logs for details."
            error_msg = QLabel(error_msg_text)
            error_msg.setAlignment(Qt.AlignCenter)
            error_msg.setWordWrap(True)
            vis_container_layout.addWidget(error_msg)
            logger.exception(f"Error creating multi-dataset 3D view:") # Log full traceback

        # Add the visualization container widget to the parent frame provided
        parent_layout.addWidget(vis_container_widget)


        return vis_container_widget # Return the main container widget

    def _set_height_factor_and_update(self, height_factor):
        """Sets the height factor and triggers a visualization update."""
        self.height_factor = height_factor
        # Trigger update for the currently visible tab
        current_index = self.notebook.currentIndex()
        self._on_tab_changed(current_index)


    def _on_tab_changed(self, index):
        """Handle tab changes to update visualizations as needed"""
        # Update visualization based on the selected tab
        logger.debug(f"Tab changed to index {index}")
        # Check if datasets exist before trying to visualize
        if not self.datasets:
             logger.debug("No datasets loaded, clearing visualizations.")
             self._clear_visualizations() # Clear plots if no data
             # Optionally show a status message in the relevant placeholder
             if index == 0: pass # File tab is fine
             elif index == 1: self._clear_hull_plot()
             elif index == 2: self._clear_segment_plot()
             elif index == 3: self._clear_triangulation_plot()
             return # Nothing more to do if no datasets

        # Datasets exist, proceed with updating the view for the new tab
        if index == 0:  # File tab
            self._visualize_all_points()
        elif index == 1:  # Hull tab
            visible_with_hulls = any(d.get('visible', True) and d.get('hull_points') is not None for d in self.datasets)
            if visible_with_hulls:
                self._visualize_all_hulls()
            else:
                # Clear the specific plot for this tab if no relevant data exists
                self._clear_hull_plot()
                self.statusBar().showMessage("No hulls computed yet. Use 'Compute Convex Hull' or right-click a dataset.")
        elif index == 2:  # Segment tab
            visible_with_segments = any(d.get('visible', True) and d.get('segments') is not None for d in self.datasets)
            if visible_with_segments:
                self._visualize_all_segments()
            else:
                self._clear_segment_plot()
                self.statusBar().showMessage("No segments computed yet. Use 'Compute Segments' after computing hulls.")
        elif index == 3:  # Triangulation tab
            visible_with_tri = any(d.get('visible', True) and d.get('triangulation_result') is not None for d in self.datasets)
            if visible_with_tri:
                self._visualize_all_triangulations()
            else:
                self._clear_triangulation_plot()
                self.statusBar().showMessage("No triangulations computed yet. Complete previous steps first.")

    # ... rest of the class methods (_get_next_color, _create_main_layout, etc.) ...

    def _get_boundary_edges(self, triangles):
        """Finds the boundary edges from a list of triangles."""
        if triangles is None or len(triangles) == 0:
            return []

        edges = set()
        boundary_edges = []

        for tri in triangles:
            # Ensure triangle vertices are sorted to make edges canonical
            tri_sorted = sorted(tri)
            # Edges: (v0, v1), (v1, v2), (v0, v2) -> use smallest index first
            current_edges = [
                tuple(sorted((tri[i], tri[(i + 1) % 3]))) for i in range(3)
            ]

            for edge in current_edges:
                if edge in edges:
                    # This edge is shared by another triangle, remove it
                    edges.remove(edge)
                else:
                    # This is the first time we see this edge
                    edges.add(edge)

        # Remaining edges in the set are boundary edges
        return list(edges) # Return as a list of tuples (v1_idx, v2_idx)

    def _create_main_layout(self):
        """Creates the main layout after widgets are initialized."""
        # This assumes self.main_layout and self.notebook exist
        if not hasattr(self, 'main_layout'):
            logger.error("_create_main_layout called before main_layout initialized.")
            self.main_layout = QVBoxLayout(self.central_widget) # Attempt recovery

        if not hasattr(self, 'notebook'):
             logger.error("_create_main_layout called before notebook initialized.")
             self.notebook = QTabWidget() # Attempt recovery
             self.main_layout.addWidget(self.notebook)

        # Ensure tabs are added after the notebook is part of the main layout
        # The setup methods already add the tabs, this function primarily ensures
        # the notebook is correctly placed in the main layout.
        # No additional widget adding needed here if setup methods are correct.
        logger.debug("Main layout structure finalized.")


    def _run_batch_computation(self, compute_type, total_items):
        """Runs a batch computation using a worker thread."""
        logger.info(f"Starting batch computation: {compute_type}")
        self._disable_compute_buttons() # Disable buttons first

        # Setup progress dialog
        self.progress_dialog = QProgressDialog(f"Computing {compute_type}...", "Cancel", 0, total_items, self)
        self.progress_dialog.setWindowTitle(f"Processing {compute_type.capitalize()}")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(False) # We will close it manually
        self.progress_dialog.setAutoReset(False) # We will reset manually
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self.cancel_computation) # Connect cancel signal
        self.progress_dialog.show()

        # Setup worker and thread
        self.worker = ComputationWorker(self) # Pass GUI instance
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.worker.dataset_finished.connect(self.handle_dataset_finished)
        self.worker.batch_finished.connect(self.handle_batch_finished)
        self.worker.error_occurred.connect(self.handle_computation_error)
        self.thread.started.connect(getattr(self.worker, f"compute_{compute_type}_batch")) # e.g., compute_hulls_batch
        self.worker.batch_finished.connect(self.thread.quit) # Quit thread when batch finishes
        self.worker.error_occurred.connect(self.thread.quit) # Also quit on error
        self.thread.finished.connect(self.worker.deleteLater) # Schedule worker deletion
        self.thread.finished.connect(self.thread.deleteLater) # Schedule thread deletion

        # Start the thread
        self.thread.start()
        logger.info(f"Worker thread started for {compute_type}.")

    def handle_dataset_finished(self, index, name, success):
        """Handles the completion of computation for a single dataset."""
        logger.debug(f"Dataset finished: Index={index}, Name='{name}', Success={success}")
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            current_value = self.progress_dialog.value()
            self.progress_dialog.setValue(current_value + 1)
            self.progress_dialog.setLabelText(f"Processed: {name} ({'OK' if success else 'Fail'})")
        QApplication.processEvents() # Keep UI responsive

    def handle_batch_finished(self, success_count, total_eligible, elapsed_time):
        """Handles the completion of a batch computation."""
        logger.info(f"Batch computation finished. Success: {success_count}/{total_eligible}. Time: {elapsed_time:.2f}s")
        # Fix: Use self.statusBar() instead of self.status_bar
        self.statusBar().showMessage(f"Batch finished: {success_count}/{total_eligible} succeeded in {elapsed_time:.2f}s.", 10000)

        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            logger.debug("Progress dialog closed.")

        if self.thread and self.thread.isRunning():
            logger.info("Signaling worker thread to stop and quit.")
            if self.worker:
                self.worker.stop() # Signal the worker loop to exit
            self.thread.quit() # Ask the thread's event loop to exit

            # Wait for the thread to finish (increased timeout to 5 seconds)
            if self.thread.wait(5000): # Wait up to 5000 ms
                logger.info("Worker thread finished gracefully.")
            else:
                # This is often a sign that the worker's run loop didn't exit cleanly.
                # Termination can be risky, so we'll log and detach.
                logger.warning("Worker thread did not finish gracefully after 5 seconds. Detaching.")
                # self.thread.terminate() # Avoid terminate() if possible, it's dangerous

        # Ensure references are cleared regardless of wait outcome
        self.thread = None
        self.worker = None
        logger.debug("Thread and worker references cleared.")

        self._enable_compute_buttons()
        self._update_visualization() # Update visualization after batch completion
        self._update_statistics()

    def handle_computation_error(self, error_message):
        """Handles errors reported by the worker thread."""
        logger.error(f"Computation error: {error_message}")
        QMessageBox.critical(self, "Computation Error", f"An error occurred during computation:\n{error_message}")

        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            logger.debug("Progress dialog closed due to error.")

        if self.thread and self.thread.isRunning():
            logger.info("Attempting to stop worker thread due to error.")
            if self.worker:
                self.worker.stop()
            self.thread.quit()

            if self.thread.wait(5000): # Wait up to 5000 ms
                logger.info("Worker thread finished after error.")
            else:
                logger.warning("Worker thread did not finish gracefully after error. Detaching.")
                # self.thread.terminate() # Avoid terminate()

        # Ensure references are cleared
        self.thread = None
        self.worker = None
        logger.debug("Thread and worker references cleared after error.")

        self._enable_compute_buttons()
        # Fix: Use self.statusBar() instead of self.status_bar
        self.statusBar().showMessage("Computation failed.", 5000)


    def cancel_computation(self):
        """Cancels the ongoing computation."""
        logger.info("Cancel computation requested by user.")
        if hasattr(self, 'worker') and self.worker:
             self.worker.stop() # Signal the worker to stop
             logger.debug("Stop signal sent to worker.")
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
             self.progress_dialog.setLabelText("Canceling computation...")
             self.progress_dialog.setEnabled(False) # Disable further interaction
             logger.debug("Progress dialog updated for cancellation.")
        # The batch_finished or error_occurred signal will handle cleanup


    def _disable_compute_buttons(self):
        """Disables the main compute buttons on all tabs."""
        logger.debug("Disabling compute buttons.")
        tabs_and_buttons = [
            (self.hull_tab, "compute_btn"),
            (self.segment_tab, "compute_btn"),
            (self.triangulation_tab, "run_btn")
        ]
        for tab_widget, button_name in tabs_and_buttons:
            if tab_widget: # Check if tab exists
                button = tab_widget.findChild(QPushButton, button_name)
                if button:
                    button.setEnabled(False)
                else:
                    logger.warning(f"Could not find button '{button_name}' in tab {tab_widget.objectName()} to disable.")
            else:
                 logger.warning(f"Tab widget not found when trying to disable button '{button_name}'.")


    def _enable_compute_buttons(self):
        """Enables the main compute buttons on all tabs."""
        logger.debug("Enabling compute buttons.")
        tabs_and_buttons = [
            (self.hull_tab, "compute_btn"),
            (self.segment_tab, "compute_btn"),
            (self.triangulation_tab, "run_btn")
        ]
        for tab_widget, button_name in tabs_and_buttons:
             if tab_widget: # Check if tab exists
                 button = tab_widget.findChild(QPushButton, button_name)
                 if button:
                     button.setEnabled(True)
                 else:
                      logger.warning(f"Could not find button '{button_name}' in tab {tab_widget.objectName()} to enable.")
             else:
                 logger.warning(f"Tab widget not found when trying to enable button '{button_name}'.")


    def closeEvent(self, event):
        """Handle closing the application, especially if a thread is running."""
        logger.debug("Close event triggered.")
        # Check if a worker thread is running
        if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
            logger.info("Close event while worker thread is running.")
            reply = QMessageBox.question(self, 'Confirm Exit',
                                       "A computation is running. Stop and exit?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                logger.info("User chose to stop and exit.")
                if hasattr(self, 'worker') and self.worker:
                    self.worker.stop() # Signal the worker to stop

                # Optionally wait briefly for the thread to stop naturally
                if not self.thread.wait(1000): # Wait 1 second
                    logger.warning("Thread did not finish quickly after stop signal. Terminating.")
                    self.thread.terminate() # Force stop if it doesn't finish
                    self.thread.wait() # Wait again after termination

                event.accept() # Allow closing
                logger.info("Application closing after stopping thread.")
            else:
                logger.info("User chose not to exit.")
                event.ignore() # Prevent closing
        else:
            logger.debug("No active thread running. Closing application.")
            # Close any standalone PyVista plotters if they exist
            if hasattr(self, 'pv_plotter') and self.pv_plotter is not None:
                 try:
                     self.pv_plotter.close()
                 except Exception as e:
                      logger.warning(f"Error closing standalone PyVista plotter: {e}")
            event.accept() # No thread running, close normally

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MeshItWorkflowGUI()
    window.show()
    sys.exit(app.exec_()) 