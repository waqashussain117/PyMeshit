"""
Simple GUI for triangulation control.

This GUI application allows interactive control of triangulation parameters
like gradient, number of boundary points, and base size.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging
import os
import sys
import tkinter as tk
from tkinter import ttk
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Triangulation-GUI")

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
    sys.exit(1)

class TriangulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Triangulation Control GUI")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        
        # Set up the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Set up the control panel (left)
        control_frame = ttk.LabelFrame(main_frame, text="Triangulation Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Add controls to the control panel
        ttk.Label(control_frame, text="Boundary Points:").grid(column=0, row=0, sticky=tk.W, pady=5)
        self.boundary_points_var = tk.IntVar(value=30)
        ttk.Spinbox(control_frame, from_=10, to=100, textvariable=self.boundary_points_var, width=5, 
                   command=self.schedule_update).grid(column=1, row=0, sticky=tk.W, pady=5)
        
        ttk.Label(control_frame, text="Domain Radius:").grid(column=0, row=1, sticky=tk.W, pady=5)
        self.radius_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(control_frame, from_=5.0, to=20.0, textvariable=self.radius_var, width=5, 
                   increment=1.0, command=self.schedule_update).grid(column=1, row=1, sticky=tk.W, pady=5)
        
        ttk.Label(control_frame, text="Base Size Factor:").grid(column=0, row=2, sticky=tk.W, pady=5)
        self.base_size_factor_var = tk.DoubleVar(value=15.0)
        ttk.Spinbox(control_frame, from_=5.0, to=30.0, textvariable=self.base_size_factor_var, width=5,
                   increment=1.0, command=self.schedule_update).grid(column=1, row=2, sticky=tk.W, pady=5)
        
        ttk.Label(control_frame, text="Gradient:").grid(column=0, row=3, sticky=tk.W, pady=5)
        self.gradient_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(control_frame, from_=1.0, to=3.0, textvariable=self.gradient_var, width=5,
                   increment=0.1, command=self.schedule_update).grid(column=1, row=3, sticky=tk.W, pady=5)
        
        ttk.Label(control_frame, text="Minimum Angle:").grid(column=0, row=4, sticky=tk.W, pady=5)
        self.min_angle_var = tk.DoubleVar(value=25.0)
        ttk.Spinbox(control_frame, from_=10.0, to=30.0, textvariable=self.min_angle_var, width=5,
                   increment=1.0, command=self.schedule_update).grid(column=1, row=4, sticky=tk.W, pady=5)
        
        # Add feature point controls
        ttk.Separator(control_frame, orient='horizontal').grid(column=0, row=5, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(control_frame, text="Feature Points:").grid(column=0, row=6, sticky=tk.W, pady=5)
        self.use_feature_points_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, variable=self.use_feature_points_var, 
                       command=self.schedule_update).grid(column=1, row=6, sticky=tk.W, pady=5)
        
        ttk.Label(control_frame, text="Number of Features:").grid(column=0, row=7, sticky=tk.W, pady=5)
        self.num_features_var = tk.IntVar(value=3)
        ttk.Spinbox(control_frame, from_=1, to=10, textvariable=self.num_features_var, width=5,
                   command=self.schedule_update).grid(column=1, row=7, sticky=tk.W, pady=5)
        
        ttk.Label(control_frame, text="Feature Size:").grid(column=0, row=8, sticky=tk.W, pady=5)
        self.feature_size_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(control_frame, from_=0.1, to=3.0, textvariable=self.feature_size_var, width=5,
                   increment=0.1, command=self.schedule_update).grid(column=1, row=8, sticky=tk.W, pady=5)
        
        # Add options for uniform triangulation
        ttk.Separator(control_frame, orient='horizontal').grid(column=0, row=9, columnspan=2, sticky=tk.EW, pady=10)
        
        ttk.Label(control_frame, text="Uniform Triangulation:").grid(column=0, row=10, sticky=tk.W, pady=5)
        self.uniform_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, variable=self.uniform_var, 
                       command=self.schedule_update).grid(column=1, row=10, sticky=tk.W, pady=5)
        
        ttk.Label(control_frame, text="Create Transition:").grid(column=0, row=11, sticky=tk.W, pady=5)
        self.create_transition_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, variable=self.create_transition_var, 
                       command=self.schedule_update).grid(column=1, row=11, sticky=tk.W, pady=5)
        
        # Add a "Run Triangulation" button
        ttk.Button(control_frame, text="Run Triangulation", 
                  command=self.run_triangulation).grid(column=0, row=12, columnspan=2, pady=20)
        
        # Add a stats display area
        self.stats_text = tk.Text(control_frame, height=8, width=30)
        self.stats_text.grid(column=0, row=13, columnspan=2, pady=10)
        self.stats_text.config(state=tk.DISABLED)
        
        # Set up the visualization panel (right)
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize data
        self.boundary_points = None
        self.segments = None
        self.triangulation_result = None
        self.feature_points = None
        self.feature_sizes = None
        self.base_size = None
        
        # Schedule the first update
        self.update_needed = False
        self.schedule_update()
        
        # Set up periodic checking for updates
        self.check_for_updates()

    def schedule_update(self):
        """Schedule an update of the triangulation"""
        self.update_needed = True
        
    def check_for_updates(self):
        """Check if an update is needed and run it if so"""
        if self.update_needed:
            self.run_triangulation()
            self.update_needed = False
        # Schedule next check
        self.root.after(500, self.check_for_updates)
    
    def create_boundary(self):
        """Create the boundary points and segments"""
        num_points = self.boundary_points_var.get()
        radius = self.radius_var.get()
        
        # Create boundary points
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        points = np.column_stack((x, y))
        
        # Create segments
        segments = np.column_stack((
            np.arange(num_points),
            np.roll(np.arange(num_points), -1)
        ))
        
        self.boundary_points = points
        self.segments = segments
    
    def create_feature_points(self):
        """Create feature points for controlling mesh density"""
        if not self.use_feature_points_var.get():
            self.feature_points = None
            self.feature_sizes = None
            return
        
        num_features = self.num_features_var.get()
        radius = self.radius_var.get()
        feature_size_base = self.feature_size_var.get()
        
        # Create feature points in interesting locations
        feature_points = []
        feature_sizes = []
        
        # Create center point
        feature_points.append([0, 0])
        feature_sizes.append(feature_size_base * 0.8)
        
        # Create other feature points at different positions
        for i in range(num_features - 1):
            angle = 2 * np.pi * i / (num_features - 1)
            # Position at around 2/3 of the radius
            distance = radius * 0.6 * (0.7 + 0.3 * np.random.random())
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            
            feature_points.append([x, y])
            # Random size variation
            feature_sizes.append(feature_size_base * (0.8 + 0.4 * np.random.random()))
        
        self.feature_points = np.array(feature_points)
        self.feature_sizes = np.array(feature_sizes) * self.base_size
    
    def run_triangulation(self):
        """Run the triangulation with current parameters"""
        # Create boundary points
        self.create_boundary()
        
        # Calculate base size
        min_coords = np.min(self.boundary_points, axis=0)
        max_coords = np.max(self.boundary_points, axis=0)
        diagonal = np.sqrt(np.sum((max_coords - min_coords) ** 2))
        base_size_factor = self.base_size_factor_var.get()
        self.base_size = diagonal / base_size_factor
        
        # Create feature points if needed
        self.create_feature_points()
        
        # Create DirectTriangleWrapper with current parameters
        gradient = self.gradient_var.get()
        min_angle = self.min_angle_var.get()
        
        wrapper = DirectTriangleWrapper(
            gradient=gradient,
            min_angle=min_angle,
            base_size=self.base_size
        )
        
        # Set feature points if available
        if self.feature_points is not None:
            wrapper.set_feature_points(self.feature_points, self.feature_sizes)
        
        # Run triangulation
        start_time = time.time()
        
        use_uniform = self.uniform_var.get()
        create_transition = self.create_transition_var.get()
        
        self.triangulation_result = wrapper.triangulate(
            points=self.boundary_points,
            segments=self.segments,
            create_feature_points=self.use_feature_points_var.get(),
            create_transition=create_transition,
            uniform=use_uniform
        )
        
        triangulation_time = time.time() - start_time
        
        # Update visualization
        self.update_visualization()
        
        # Update stats
        vertices = self.triangulation_result['vertices']
        triangles = self.triangulation_result['triangles']
        
        # Calculate mesh statistics
        edge_lengths = []
        for tri in triangles:
            v1, v2, v3 = vertices[tri]
            edge_lengths.extend([
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v3 - v2),
                np.linalg.norm(v1 - v3)
            ])
        
        mean_edge = np.mean(edge_lengths)
        std_edge = np.std(edge_lengths)
        
        # Update stats display
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"Triangles: {len(triangles)}\n")
        self.stats_text.insert(tk.END, f"Vertices: {len(vertices)}\n")
        self.stats_text.insert(tk.END, f"Base size: {self.base_size:.4f}\n")
        self.stats_text.insert(tk.END, f"Mean edge: {mean_edge:.4f}\n")
        self.stats_text.insert(tk.END, f"Edge std: {std_edge:.4f}\n")
        self.stats_text.insert(tk.END, f"Uniformity: {std_edge/mean_edge:.4f}\n")
        self.stats_text.insert(tk.END, f"Time: {triangulation_time:.2f}s\n")
        self.stats_text.config(state=tk.DISABLED)
        
    def update_visualization(self):
        """Update the visualization with current triangulation"""
        if self.triangulation_result is None:
            return
        
        # Clear previous plot
        self.ax.clear()
        
        # Get triangulation data
        vertices = self.triangulation_result['vertices']
        triangles = self.triangulation_result['triangles']
        
        # Plot triangulation
        self.ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'k-', alpha=0.5)
        
        # Plot boundary
        self.ax.plot(np.append(self.boundary_points[:, 0], self.boundary_points[0, 0]),
                   np.append(self.boundary_points[:, 1], self.boundary_points[0, 1]),
                   'r-', linewidth=2, label='Boundary')
        
        # Plot feature points if available
        if self.feature_points is not None:
            # Draw feature points with circles proportional to size
            self.ax.scatter(self.feature_points[:, 0], self.feature_points[:, 1], 
                          c='orange', s=100, label='Feature Points')
            
            # Draw circles showing feature influence
            for i, (point, size) in enumerate(zip(self.feature_points, self.feature_sizes)):
                circle = plt.Circle((point[0], point[1]), size, fill=False, 
                                   color='orange', linestyle='--', alpha=0.5)
                self.ax.add_patch(circle)
        
        # Set up plot
        self.ax.set_title(f'Triangulation ({len(triangles)} triangles)', fontsize=14)
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.3)
        if self.feature_points is not None:
            self.ax.legend()
        
        # Update canvas
        self.canvas.draw()

if __name__ == "__main__":
    # Create the root window
    root = tk.Tk()
    
    # Create the application
    app = TriangulationGUI(root)
    
    # Run the application
    root.mainloop() 