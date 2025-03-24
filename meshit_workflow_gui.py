"""
MeshIt Workflow GUI

This application provides a graphical interface for the complete MeshIt workflow,
including file loading, convex hull computation, segmentation, triangulation, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import logging
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time

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

class MeshItWorkflowGUI:
    def __init__(self, root=None):
        # Create root window if none provided
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
            
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for each workflow step
        self.file_tab = ttk.Frame(self.notebook)
        self.hull_tab = ttk.Frame(self.notebook)
        self.segment_tab = ttk.Frame(self.notebook)
        self.triangulation_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.file_tab, text="1. Load Data")
        self.notebook.add(self.hull_tab, text="2. Convex Hull")
        self.notebook.add(self.segment_tab, text="3. Segmentation")
        self.notebook.add(self.triangulation_tab, text="4. Triangulation")
        
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
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_file_tab(self):
        """Sets up the file loading tab with controls and visualization area"""
        # Create frames for controls and visualization
        control_frame = ttk.Frame(self.file_tab)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # -- File Loading Controls --
        file_frame = ttk.LabelFrame(control_frame, text="Data Import")
        file_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Buttons for loading files
        load_btn = ttk.Button(file_frame, text="Load File", command=self.load_file)
        load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Test data generation
        test_btn = ttk.Button(file_frame, text="Generate Test Data", 
                            command=self.generate_test_data)
        test_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # -- Statistics --
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Number of points
        self.points_var = tk.StringVar(value="Points: 0")
        points_label = ttk.Label(stats_frame, textvariable=self.points_var)
        points_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Bounding box info
        self.bounds_var = tk.StringVar(value="Bounds: N/A")
        bounds_label = ttk.Label(stats_frame, textvariable=self.bounds_var)
        bounds_label.pack(fill=tk.X, padx=5, pady=2)
        
        # -- Visualization Area --
        viz_frame = ttk.LabelFrame(self.file_tab, text="Point Cloud Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create container frame for visualization
        self.file_viz_frame = ttk.Frame(viz_frame)
        self.file_viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Load data to visualize points in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nLoad data to visualize points in 2D."
            
        self.file_viz_placeholder = ttk.Label(self.file_viz_frame, 
                                             text=placeholder_text,
                                             anchor="center", justify="center")
        self.file_viz_placeholder.pack(expand=True)
        
    def _setup_hull_tab(self):
        """Sets up the convex hull tab with controls and visualization area"""
        # Create frames for controls and visualization
        control_frame = ttk.Frame(self.hull_tab)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # -- Hull Controls --
        hull_controls = ttk.LabelFrame(control_frame, text="Hull Controls")
        hull_controls.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Compute hull button
        compute_btn = ttk.Button(hull_controls, text="Compute Convex Hull", 
                                command=self.compute_hull)
        compute_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # -- Statistics --
        stats_frame = ttk.LabelFrame(control_frame, text="Hull Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Hull points count
        self.hull_points_var = tk.StringVar(value="Hull vertices: 0")
        hull_points_label = ttk.Label(stats_frame, textvariable=self.hull_points_var)
        hull_points_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Hull area
        self.hull_area_var = tk.StringVar(value="Hull area: 0.0")
        hull_area_label = ttk.Label(stats_frame, textvariable=self.hull_area_var)
        hull_area_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=15)
        
        prev_btn = ttk.Button(nav_frame, text="← Previous", 
                             command=lambda: self.notebook.select(0))
        prev_btn.pack(side=tk.LEFT)
        
        next_btn = ttk.Button(nav_frame, text="Next →", 
                             command=lambda: self.notebook.select(2))
        next_btn.pack(side=tk.RIGHT)
        
        # -- Visualization Area --
        viz_frame = ttk.LabelFrame(self.hull_tab, text="Convex Hull Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create container frame for visualization
        self.hull_viz_frame = ttk.Frame(viz_frame)
        self.hull_viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Compute convex hull to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute convex hull to visualize in 2D."
            
        self.hull_viz_placeholder = ttk.Label(self.hull_viz_frame, 
                                             text=placeholder_text,
                                             anchor="center", justify="center")
        self.hull_viz_placeholder.pack(expand=True)
    
    def _setup_segment_tab(self):
        """Sets up the segmentation tab with controls and visualization area"""
        # Create frames for controls and visualization
        control_frame = ttk.Frame(self.segment_tab)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # -- Segmentation Controls --
        segment_controls = ttk.LabelFrame(control_frame, text="Segmentation Controls")
        segment_controls.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Segment length control
        length_frame = ttk.Frame(segment_controls)
        length_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(length_frame, text="Segment Length:").pack(side=tk.LEFT)
        self.segment_length_var = tk.StringVar(value="1.0")
        ttk.Entry(length_frame, textvariable=self.segment_length_var, width=8).pack(side=tk.RIGHT)
        
        # Segment density control
        density_frame = ttk.Frame(segment_controls)
        density_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(density_frame, text="Density:").pack(side=tk.LEFT)
        self.segment_density_var = tk.DoubleVar(value=1.0)
        ttk.Scale(density_frame, variable=self.segment_density_var, 
                 from_=0.5, to=2.0, orient=tk.HORIZONTAL).pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Compute segments button
        compute_btn = ttk.Button(segment_controls, text="Compute Segmentation", 
                                command=self.compute_segments)
        compute_btn.pack(fill=tk.X, padx=5, pady=10)
        
        # -- Statistics --
        stats_frame = ttk.LabelFrame(control_frame, text="Segment Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Number of segments
        self.num_segments_var = tk.StringVar(value="Segments: 0")
        segments_label = ttk.Label(stats_frame, textvariable=self.num_segments_var)
        segments_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Average segment length
        self.avg_segment_length_var = tk.StringVar(value="Avg length: 0.0")
        avg_length_label = ttk.Label(stats_frame, textvariable=self.avg_segment_length_var)
        avg_length_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=15)
        
        prev_btn = ttk.Button(nav_frame, text="← Previous", 
                             command=lambda: self.notebook.select(1))
        prev_btn.pack(side=tk.LEFT)
        
        next_btn = ttk.Button(nav_frame, text="Next →", 
                             command=lambda: self.notebook.select(3))
        next_btn.pack(side=tk.RIGHT)
        
        # -- Visualization Area --
        viz_frame = ttk.LabelFrame(self.segment_tab, text="Segmentation Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create container frame for visualization
        self.segment_viz_frame = ttk.Frame(viz_frame)
        self.segment_viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Compute segmentation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute segmentation to visualize in 2D."
            
        self.segment_viz_placeholder = ttk.Label(self.segment_viz_frame, 
                                               text=placeholder_text,
                                               anchor="center", justify="center")
        self.segment_viz_placeholder.pack(expand=True)
    
    def _setup_triangulation_tab(self):
        """Sets up the triangulation tab with controls and visualization area"""
        # Create frames for controls and visualization
        control_frame = ttk.Frame(self.triangulation_tab)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # -- Triangulation Controls --
        tri_controls = ttk.LabelFrame(control_frame, text="Triangulation Controls")
        tri_controls.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Mesh density controls
        density_frame = ttk.Frame(tri_controls)
        density_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(density_frame, text="Mesh Density:").pack(side=tk.LEFT)
        self.base_size_factor_var = tk.DoubleVar(value=15.0)
        ttk.Spinbox(density_frame, from_=5.0, to=30.0, textvariable=self.base_size_factor_var,
                   increment=1.0, width=5).pack(side=tk.RIGHT)
        
        # Mesh quality controls
        quality_frame = ttk.LabelFrame(tri_controls, text="Quality Settings")
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Gradient
        gradient_frame = ttk.Frame(quality_frame)
        gradient_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(gradient_frame, text="Gradient:").pack(side=tk.LEFT)
        self.gradient_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(gradient_frame, from_=1.0, to=3.0, textvariable=self.gradient_var,
                   increment=0.1, width=5).pack(side=tk.RIGHT)
        
        # Min angle
        angle_frame = ttk.Frame(quality_frame)
        angle_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(angle_frame, text="Min Angle:").pack(side=tk.LEFT)
        self.min_angle_var = tk.DoubleVar(value=25.0)
        ttk.Spinbox(angle_frame, from_=10.0, to=30.0, textvariable=self.min_angle_var,
                   increment=1.0, width=5).pack(side=tk.RIGHT)
        
        # Uniform triangulation
        uniform_frame = ttk.Frame(quality_frame)
        uniform_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(uniform_frame, text="Uniform:").pack(side=tk.LEFT)
        self.uniform_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(uniform_frame, variable=self.uniform_var).pack(side=tk.RIGHT)
        
        # Feature points controls
        feature_frame = ttk.LabelFrame(tri_controls, text="Feature Points")
        feature_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Use features
        use_features_frame = ttk.Frame(feature_frame)
        use_features_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(use_features_frame, text="Use Features:").pack(side=tk.LEFT)
        self.use_feature_points_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(use_features_frame, variable=self.use_feature_points_var).pack(side=tk.RIGHT)
        
        # Number of features
        num_features_frame = ttk.Frame(feature_frame)
        num_features_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(num_features_frame, text="Count:").pack(side=tk.LEFT)
        self.num_features_var = tk.IntVar(value=3)
        ttk.Spinbox(num_features_frame, from_=1, to=10, textvariable=self.num_features_var,
                   width=5).pack(side=tk.RIGHT)
        
        # Feature size
        feature_size_frame = ttk.Frame(feature_frame)
        feature_size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(feature_size_frame, text="Size:").pack(side=tk.LEFT)
        self.feature_size_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(feature_size_frame, from_=0.1, to=3.0, textvariable=self.feature_size_var,
                   increment=0.1, width=5).pack(side=tk.RIGHT)
        
        # 3D visualization settings
        viz3d_frame = ttk.LabelFrame(tri_controls, text="3D Settings")
        viz3d_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Height scale
        height_frame = ttk.Frame(viz3d_frame)
        height_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(height_frame, text="Height Scale:").pack(side=tk.LEFT)
        self.height_factor_var = tk.DoubleVar(value=0.2)
        scale = ttk.Scale(height_frame, variable=self.height_factor_var, from_=0.0, to=1.0,
                         orient=tk.HORIZONTAL)
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Run triangulation button
        ttk.Button(tri_controls, text="Run Triangulation", 
                  command=self.run_triangulation).pack(fill=tk.X, padx=5, pady=10)
        
        # -- Statistics --
        stats_frame = ttk.LabelFrame(control_frame, text="Triangulation Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        
        # Number of triangles
        self.num_triangles_var = tk.StringVar(value="Triangles: 0")
        triangles_label = ttk.Label(stats_frame, textvariable=self.num_triangles_var)
        triangles_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Number of vertices
        self.num_vertices_var = tk.StringVar(value="Vertices: 0")
        vertices_label = ttk.Label(stats_frame, textvariable=self.num_vertices_var)
        vertices_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Mean edge length
        self.mean_edge_var = tk.StringVar(value="Mean edge: 0.0")
        edge_label = ttk.Label(stats_frame, textvariable=self.mean_edge_var)
        edge_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Edge uniformity
        self.uniformity_var = tk.StringVar(value="Uniformity: 0.0")
        uniformity_label = ttk.Label(stats_frame, textvariable=self.uniformity_var)
        uniformity_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Export button
        ttk.Button(control_frame, text="Export Results...", 
                  command=self.export_results).pack(fill=tk.X, padx=5, pady=10)
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        prev_btn = ttk.Button(nav_frame, text="← Previous", 
                             command=lambda: self.notebook.select(2))
        prev_btn.pack(side=tk.LEFT)
        
        # -- Visualization Area --
        viz_frame = ttk.LabelFrame(self.triangulation_tab, text="Triangulation Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create container frame for visualization
        self.tri_viz_frame = ttk.Frame(viz_frame)
        self.tri_viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial visualization panel will show placeholder message
        if HAVE_PYVISTA:
            placeholder_text = "Run triangulation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nRun triangulation to visualize in 2D."
            
        self.tri_viz_placeholder = ttk.Label(self.tri_viz_frame, 
                                           text=placeholder_text,
                                           anchor="center", justify="center")
        self.tri_viz_placeholder.pack(expand=True)
    
    # Event handlers - placeholder implementations
    def load_file(self):
        """Load data from a file"""
        self.status_var.set("Loading file...")
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select a point data file",
            filetypes=[
                ("Text files", "*.txt"),
                ("Data files", "*.dat"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            self.status_var.set("File loading canceled")
            return
            
        # Update status with file path
        self.status_var.set(f"Loading file: {os.path.basename(file_path)}...")
        
        try:
            # Try to read the file
            points = self._read_point_file(file_path)
            
            if points is not None and len(points) > 0:
                self.points = points
                
                # Visualize the loaded points
                self._plot_points(points)
                
                self.status_var.set(f"Successfully loaded {len(points)} points")
                
                # Clear any previous results from later steps
                self.hull_points = None
                self.segments = None
                self.triangulation_result = None
                
                # Update other views
                self._clear_hull_plot()
                self._clear_segment_plot()
                self._clear_triangulation_plot()
            else:
                self.status_var.set("Error: No valid points found in file")
                messagebox.showerror("Error", "No valid points found in file")
        
        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
            logger.error(f"Error loading file: {str(e)}")
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
    
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
            messagebox.showerror("Error", f"Error reading file: {str(e)}")
            return None
    
    def generate_test_data(self):
        """Generate test data"""
        self.status_var.set("Generating test data...")
        
        # Open a dialog to select the type of test data
        test_data_types = {
            "Random Points": self._generate_random_points,
            "Circle": self._generate_circle_points,
            "Square": self._generate_square_points,
            "Complex Shape": self._generate_complex_shape
        }
        
        # Simple dialog for data type
        dialog = tk.Toplevel(self.root)
        dialog.title("Generate Test Data")
        dialog.geometry("300x250")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select data type:").pack(pady=5)
        
        data_type_var = tk.StringVar(value="Random Points")
        for data_type in test_data_types.keys():
            ttk.Radiobutton(dialog, text=data_type, variable=data_type_var, 
                           value=data_type).pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Label(dialog, text="Number of points:").pack(pady=5)
        num_points_var = tk.IntVar(value=100)
        ttk.Spinbox(dialog, from_=10, to=1000, textvariable=num_points_var, 
                   width=5).pack(pady=5)
        
        def on_generate():
            data_type = data_type_var.get()
            num_points = num_points_var.get()
            dialog.destroy()
            
            # Generate the selected data type
            if data_type in test_data_types:
                points = test_data_types[data_type](num_points)
                
                if points is not None:
                    self.points = points
                    
                    # Visualize the generated points
                    self._plot_points(points)
                    
                    self.status_var.set(f"Generated {len(points)} {data_type} points")
                    
                    # Clear any previous results from later steps
                    self.hull_points = None
                    self.segments = None
                    self.triangulation_result = None
                    
                    # Update other views
                    self._clear_hull_plot()
                    self._clear_segment_plot()
                    self._clear_triangulation_plot()
        
        ttk.Button(dialog, text="Generate", command=on_generate).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)
        
        # Center the dialog on the parent window
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Wait for the dialog to be closed
        self.root.wait_window(dialog)
    
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
    
    # Plotting utilities
    def _plot_points(self, points):
        """Visualize the loaded points using 3D visualization instead of matplotlib
        
        Args:
            points: 2D points to visualize
        """
        if points is None or len(points) == 0:
            return
            
        # Update statistics
        self.points_var.set(f"Points: {len(points)}")
        
        # Calculate bounds
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        self.bounds_var.set(f"Bounds: X[{min_x:.2f},{max_x:.2f}] Y[{min_y:.2f},{max_y:.2f}]")
        
        # Clear existing visualization
        for widget in self.file_viz_frame.winfo_children():
            widget.destroy()
            
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
            canvas = FigureCanvasTkAgg(fig, master=self.file_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(self.file_viz_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
    
    def _clear_hull_plot(self):
        """Clear the hull plot"""
        # Clear existing visualization
        for widget in self.hull_viz_frame.winfo_children():
            widget.destroy()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Compute convex hull to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute convex hull to visualize in 2D."
            
        self.hull_viz_placeholder = ttk.Label(self.hull_viz_frame, 
                                             text=placeholder_text,
                                             anchor="center", justify="center")
        self.hull_viz_placeholder.pack(expand=True)
        
        # Reset hull info
        self.hull_points_var.set("Hull vertices: 0")
        self.hull_area_var.set("Hull area: 0.0")
    
    def _clear_segment_plot(self):
        """Clear the segmentation plot"""
        # Clear existing visualization
        for widget in self.segment_viz_frame.winfo_children():
            widget.destroy()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Compute segmentation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nCompute segmentation to visualize in 2D."
            
        self.segment_viz_placeholder = ttk.Label(self.segment_viz_frame, 
                                               text=placeholder_text,
                                               anchor="center", justify="center")
        self.segment_viz_placeholder.pack(expand=True)
        
        # Reset segment info
        self.num_segments_var.set("Segments: 0")
        self.avg_segment_length_var.set("Avg length: 0.0")
    
    def _clear_triangulation_plot(self):
        """Clear the triangulation plot"""
        # Clear existing visualization
        for widget in self.tri_viz_frame.winfo_children():
            widget.destroy()
            
        # Reset placeholder
        if HAVE_PYVISTA:
            placeholder_text = "Run triangulation to visualize in 3D"
        else:
            placeholder_text = "PyVista not available. Install PyVista for 3D visualization.\nRun triangulation to visualize in 2D."
            
        self.tri_viz_placeholder = ttk.Label(self.tri_viz_frame, 
                                           text=placeholder_text,
                                           anchor="center", justify="center")
        self.tri_viz_placeholder.pack(expand=True)
        
        # Reset triangulation info
        self.num_triangles_var.set("Triangles: 0")
        self.num_vertices_var.set("Vertices: 0")
        self.mean_edge_var.set("Mean edge: 0.0")
        self.uniformity_var.set("Uniformity: 0.0")
    
    def compute_hull(self):
        """Compute the convex hull of the loaded points"""
        self.status_var.set("Computing convex hull...")
        
        if self.points is None or len(self.points) < 3:
            self.status_var.set("Error: Need at least 3 points to compute hull")
            messagebox.showerror("Error", "Need at least 3 points to compute hull")
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
            self.hull_points_var.set(f"Hull vertices: {len(hull.vertices)}")
            self.hull_area_var.set(f"Hull area: {hull_area:.2f}")
            
            # Visualize the hull
            self._plot_hull(self.points, self.hull_points)
            
            self.status_var.set(f"Computed convex hull with {len(hull.vertices)} vertices")
            
            # Clear any previous results from later steps
            self.segments = None
            self.triangulation_result = None
            
            # Update other views
            self._clear_segment_plot()
            self._clear_triangulation_plot()
            
            # Auto-switch to the hull tab
            self.notebook.select(1)  # Select hull tab (index 1)
            
        except Exception as e:
            self.status_var.set(f"Error computing hull: {str(e)}")
            logger.error(f"Error computing hull: {str(e)}")
            messagebox.showerror("Error", f"Error computing hull: {str(e)}")
    
    def _plot_hull(self, points, hull_points):
        """Visualize the hull using 3D visualization instead of matplotlib
        
        Args:
            points: All input points
            hull_points: Points of the convex hull
        """
        if points is None or len(points) == 0 or hull_points is None or len(hull_points) == 0:
            return
            
        # Update statistics
        self.hull_points_var.set(f"Hull vertices: {len(hull_points)-1}")  # -1 for the closing point
        
        # Calculate hull area (approximate)
        area = 0.0
        for i in range(len(hull_points)-1):
            x1, y1 = hull_points[i]
            x2, y2 = hull_points[i+1]
            area += x1*y2 - x2*y1
        area = abs(area) / 2.0
        self.hull_area_var.set(f"Hull area: {area:.2f}")
        
        # Clear existing visualization
        for widget in self.hull_viz_frame.winfo_children():
            widget.destroy()
            
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
            canvas = FigureCanvasTkAgg(fig, master=self.hull_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(self.segment_viz_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
    
    def compute_segments(self):
        """Compute the segmentation of the convex hull"""
        self.status_var.set("Computing segments...")
        
        if self.hull_points is None or len(self.hull_points) < 4:  # 3 vertices + 1 closing point
            self.status_var.set("Error: Compute convex hull first")
            messagebox.showerror("Error", "Please compute the convex hull first")
            return
        
        # Get the segmentation parameters from the UI
        try:
            segment_length = float(self.segment_length_var.get())
            if segment_length <= 0:
                segment_length = 1.0
                self.segment_length_var.set("1.0")
        except ValueError:
            segment_length = 1.0
            self.segment_length_var.set("1.0")
        
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
                self.segment_length_var.set(f"{segment_length:.2f}")
            
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
            self.num_segments_var.set(f"Segments: {len(segments)}")
            self.avg_segment_length_var.set(f"Avg length: {avg_segment_length:.2f}")
            
            # Visualize the segments
            self._plot_segments(self.points, self.hull_points, self.segments)
            
            self.status_var.set(f"Computed {len(segments)} segments, avg length: {avg_segment_length:.2f}")
            
            # Clear any previous results from later steps
            self.triangulation_result = None
            
            # Update other views
            self._clear_triangulation_plot()
            
            # Auto-switch to the segments tab
            self.notebook.select(2)  # Select segments tab (index 2)
            
        except Exception as e:
            self.status_var.set(f"Error computing segments: {str(e)}")
            logger.error(f"Error computing segments: {str(e)}")
            messagebox.showerror("Error", f"Error computing segments: {str(e)}")
            
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
        self.num_segments_var.set(f"Segments: {len(segments)}")
        
        # Calculate average segment length
        segment_lengths = [np.linalg.norm(segment[1] - segment[0]) for segment in segments]
        avg_length = np.mean(segment_lengths)
        self.avg_segment_length_var.set(f"Avg length: {avg_length:.2f}")
        
        # Clear existing visualization
        for widget in self.segment_viz_frame.winfo_children():
            widget.destroy()
            
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
            canvas = FigureCanvasTkAgg(fig, master=self.segment_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(self.segment_viz_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
    
    def run_triangulation(self):
        """Run triangulation on the segments and points"""
        self.status_var.set("Running triangulation...")
        
        if self.segments is None or len(self.segments) < 3:
            self.status_var.set("Error: Compute segments first")
            messagebox.showerror("Error", "Please compute segments first")
            return
        
        # Get triangulation parameters from UI
        gradient = self.gradient_var.get()
        min_angle = self.min_angle_var.get()
        base_size_factor = self.base_size_factor_var.get()
        uniform = self.uniform_var.get()
        use_feature_points = self.use_feature_points_var.get()
        
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
            self.status_var.set(f"Running triangulation with base size {base_size:.2f}...")
            
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
            
            self.status_var.set(f"Completed triangulation: {len(triangles)} triangles, {len(vertices)} vertices in {elapsed_time:.2f}s")
            
            # Auto-switch to the triangulation tab
            self.notebook.select(3)  # Select triangulation tab (index 3)
            
        except Exception as e:
            self.status_var.set(f"Error in triangulation: {str(e)}")
            logger.error(f"Error during triangulation: {str(e)}")
            messagebox.showerror("Error", f"Error in triangulation: {str(e)}")
    
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
        self.num_triangles_var.set(f"Triangles: {len(triangles)}")
        self.num_vertices_var.set(f"Vertices: {len(vertices)}")
        
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
        
        self.mean_edge_var.set(f"Mean edge: {mean_edge:.4f}")
        self.uniformity_var.set(f"Uniformity: {uniformity:.4f}")
        
        # Clear existing visualization
        for widget in self.tri_viz_frame.winfo_children():
            widget.destroy()

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
            canvas = FigureCanvasTkAgg(fig, master=self.tri_viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar_frame = ttk.Frame(self.tri_viz_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
    
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
            msg_frame = ttk.Frame(parent_frame)
            msg_frame.pack(fill=tk.BOTH, expand=True)
            
            msg = ttk.Label(msg_frame, text="PyVista not installed.\n"
                           "Please install PyVista for 3D visualization.")
            msg.pack(expand=True)
            return msg_frame
        
        # Create a frame for the visualization
        vis_frame = ttk.Frame(parent_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True)
        
        # Close previous plotter if it exists
        if self.current_plotter is not None:
            try:
                self.current_plotter.close()
            except:
                pass
        
        # Add button to open a separate interactive window
        ttk.Button(vis_frame, text="Open 3D Interactive View", 
                  command=lambda: self._show_interactive_3d(
                      points=points, 
                      point_colors=point_colors, 
                      lines=lines, 
                      triangles=triangles, 
                      title=title
                  )).pack(pady=5)
        
        # Create a placeholder for the 3D visualization
        preview_frame = ttk.LabelFrame(vis_frame, text="3D Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add a label with info about the visualization
        vis_info = f"{title}\n"
        vis_info += f"Points: {len(points)}\n"
        if lines is not None and len(lines) > 0:
            vis_info += f"Lines: {len(lines)}\n"
        if triangles is not None and len(triangles) > 0:
            vis_info += f"Triangles: {len(triangles)}\n"
            
        info_label = ttk.Label(preview_frame, text=vis_info, justify="center")
        info_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        note_label = ttk.Label(preview_frame, 
                             text="Click 'Open 3D Interactive View' for full 3D visualization",
                             justify="center", font=("Arial", 10, "italic"))
        note_label.pack(fill=tk.X, pady=10)
        
        return vis_frame
    
    def show_3d_view(self):
        """Show 3D view of the triangulation in a separate window"""
        if not self.view_3d_enabled:
            messagebox.showinfo("PyVista Not Available", 
                             "PyVista is not available. Please install PyVista for 3D visualization.")
            return
            
        if not hasattr(self, 'triangulation_result') or self.triangulation_result is None:
            messagebox.showinfo("No Triangulation", 
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
    
    def _show_interactive_3d(self, points, point_colors=None, lines=None, triangles=None,
                           feature_points=None, title="MeshIt 3D View"):
        """Open an interactive 3D visualization window"""
        if not self.view_3d_enabled:
            messagebox.showinfo("PyVista Not Available", 
                               "PyVista is not available. Please install PyVista for 3D visualization.")
            return
            
        if points is None or len(points) == 0:
            messagebox.showinfo("No Data", 
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
        
    def export_results(self):
        """Export the triangulation results to a file"""
        self.status_var.set("Exporting results...")
        
        if self.triangulation_result is None:
            self.status_var.set("Error: No triangulation results to export")
            messagebox.showerror("Error", "No triangulation results to export")
            return
        
        # Create export options dialog
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Export Options")
        export_dialog.geometry("400x300")
        export_dialog.transient(self.root)
        export_dialog.grab_set()
        
        # Export format frame
        format_frame = ttk.LabelFrame(export_dialog, text="Export Format")
        format_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        format_var = tk.StringVar(value="obj")
        ttk.Radiobutton(format_frame, text="OBJ Format (3D objects)", 
                       variable=format_var, value="obj").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(format_frame, text="PLY Format (3D mesh)", 
                       variable=format_var, value="ply").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(format_frame, text="STL Format (3D printing)", 
                       variable=format_var, value="stl").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(format_frame, text="CSV Format (raw data)", 
                       variable=format_var, value="csv").grid(row=3, column=0, sticky=tk.W, pady=2)
        
        # 3D export options (enabled only if PyVista is available)
        options_frame = ttk.LabelFrame(export_dialog, text="3D Export Options")
        options_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        export_3d_var = tk.BooleanVar(value=HAVE_PYVISTA)
        ttk.Checkbutton(options_frame, text="Export as 3D Surface", 
                       variable=export_3d_var,
                       state=tk.NORMAL if HAVE_PYVISTA else tk.DISABLED).grid(row=0, column=0, 
                                                                          sticky=tk.W, pady=2)
        
        ttk.Label(options_frame, text="Height Scale:").grid(row=1, column=0, sticky=tk.W, pady=2)
        height_scale_var = tk.DoubleVar(value=self.height_factor_var.get())
        ttk.Scale(options_frame, variable=height_scale_var, from_=0.0, to=1.0, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.W + tk.E, pady=2)
                
        # Export function
        def do_export():
            # Get export format and 3D options
            export_format = format_var.get()
            export_3d = export_3d_var.get() and HAVE_PYVISTA
            height_scale = height_scale_var.get()
            
            # Create a file save dialog
            file_path = filedialog.asksaveasfilename(
                title="Save triangulation results",
                filetypes=[
                    ("OBJ files", "*.obj"),
                    ("PLY files", "*.ply"),
                    ("STL files", "*.stl"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ],
                defaultextension=f".{export_format}"
            )
            
            if not file_path:
                export_dialog.destroy()
                self.status_var.set("Export canceled")
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
                            self.status_var.set(f"Exported 3D OBJ file to {file_path}")
                        elif export_format == "ply":
                            mesh.save(file_path)
                            self.status_var.set(f"Exported 3D PLY file to {file_path}")
                        elif export_format == "stl":
                            mesh.save(file_path)
                            self.status_var.set(f"Exported 3D STL file to {file_path}")
                        elif export_format == "csv":
                            # For CSV, fall back to standard CSV export
                            self._export_csv(file_path, points_3d, triangles)
                            self.status_var.set(f"Exported 3D CSV file to {file_path}")
                    except Exception as e:
                        self.status_var.set(f"Error creating 3D export: {str(e)}")
                        logger.error(f"Error creating 3D export: {str(e)}")
                        messagebox.showerror("Export Error", f"Error creating 3D export: {str(e)}")
                else:
                    # Standard 2D format exports
                    if export_format == "obj":
                        self._export_obj(file_path, vertices, triangles)
                        self.status_var.set(f"Exported 2D OBJ file to {file_path}")
                    elif export_format == "ply":
                        self._export_ply(file_path, vertices, triangles)
                        self.status_var.set(f"Exported 2D PLY file to {file_path}")
                    elif export_format == "stl":
                        self._export_stl(file_path, vertices, triangles)
                        self.status_var.set(f"Exported 2D STL file to {file_path}")
                    elif export_format == "csv":
                        self._export_csv(file_path, vertices, triangles)
                        self.status_var.set(f"Exported 2D CSV file to {file_path}")
                
                export_dialog.destroy()
                
            except Exception as e:
                self.status_var.set(f"Error exporting results: {str(e)}")
                logger.error(f"Error exporting results: {str(e)}")
                messagebox.showerror("Error", f"Error exporting results: {str(e)}")
                export_dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(export_dialog)
        button_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Export", command=do_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=export_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Center dialog on parent
        export_dialog.update_idletasks()
        width = export_dialog.winfo_width()
        height = export_dialog.winfo_height()
        x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        export_dialog.geometry(f"+{x}+{y}")
    
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

    def run(self):
        """Run the application"""
        # Set window title to include 3D visualization status
        if self.view_3d_enabled:
            title = "MeshIt Workflow GUI - 3D Visualization Enabled"
        else:
            title = "MeshIt Workflow GUI - 3D Visualization Disabled (Install PyVista for 3D)"
            
        self.root.title(title)
        
        # Create a menu bar
        menubar = tk.Menu(self.root)
        
        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Data", command=self.load_file)
        filemenu.add_command(label="Generate Test Data", command=self.generate_test_data)
        filemenu.add_separator()
        filemenu.add_command(label="Export Results", command=self.export_results)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        
        # Visualization menu
        vizmenu = tk.Menu(menubar, tearoff=0)
        if self.view_3d_enabled:
            vizmenu.add_command(label="Show 3D View", command=self.show_3d_view)
            vizmenu.add_separator()
            
            # 3D height factor submenu
            heightmenu = tk.Menu(vizmenu, tearoff=0)
            for height in [0.0, 0.1, 0.2, 0.5, 1.0]:
                heightmenu.add_command(
                    label=f"Height Factor: {height}", 
                    command=lambda h=height: self._set_height_factor(h)
                )
            vizmenu.add_cascade(label="3D Height Factor", menu=heightmenu)
        else:
            vizmenu.add_command(label="3D Visualization Disabled", state=tk.DISABLED)
            vizmenu.add_command(label="Install PyVista for 3D features", state=tk.DISABLED)
        
        menubar.add_cascade(label="Visualization", menu=vizmenu)
        
        # Help menu
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        # Set the menu bar
        self.root.config(menu=menubar)
        
        # Start the application
        self.root.mainloop()
        
    def _set_height_factor(self, height):
        """Set the height factor for 3D visualization"""
        self.height_factor = height
        self.height_factor_var.set(height)
        
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
        
        messagebox.showinfo("About MeshIt Workflow GUI", about_text)

# Main entry point
if __name__ == "__main__":
    app = MeshItWorkflowGUI()
    app.run() 