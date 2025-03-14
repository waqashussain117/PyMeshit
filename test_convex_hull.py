import meshit
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import pyvista as pv
import os

def save_points_to_vtk(points, filename):
    """Save points to a VTK file."""
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Points data\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        # Add vertices (each point is a vertex)
        f.write(f"VERTICES {len(points)} {len(points) * 2}\n")
        for i in range(len(points)):
            f.write(f"1 {i}\n")
    
    print(f"Points saved to VTK file: {filename}")

def save_convex_hull_to_vtk(points, hull, filename):
    """Save convex hull to a VTK file."""
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Convex Hull\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        # Write the faces of the convex hull
        total_size = len(hull.simplices) + sum(len(simplex) for simplex in hull.simplices)
        f.write(f"POLYGONS {len(hull.simplices)} {total_size}\n")
        
        for simplex in hull.simplices:
            f.write(f"{len(simplex)}")
            for idx in simplex:
                f.write(f" {idx}")
            f.write("\n")
    
    print(f"Convex hull saved to VTK file: {filename}")

def visualize_with_pyvista(points, hull_points=None, hull=None, title="PyVista Visualization"):
    """
    Interactive visualization using PyVista.
    
    Parameters:
    -----------
    points : array-like
        Original scattered points
    hull_points : array-like, optional
        Points of the convex hull
    hull : scipy.spatial.ConvexHull, optional
        ConvexHull object with simplices
    title : str
        Title for the visualization
    """
    # Create a plotter
    plotter = pv.Plotter()
    plotter.set_background('white')
    plotter.add_title(title, font_size=16)
    
    # Add original points as a point cloud
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud, color='blue', point_size=10, render_points_as_spheres=True, label="Original Points")
    
    # Add convex hull if provided
    if hull_points is not None and hull is not None:
        # Create faces array for pyvista
        faces = []
        for simplex in hull.simplices:
            faces.append(len(simplex))
            for idx in simplex:
                faces.append(idx)
        
        # Create a polydata for the hull
        hull_polydata = pv.PolyData(hull_points, faces=faces)
        plotter.add_mesh(hull_polydata, color='red', opacity=0.5, label="Convex Hull")
    
    # Add axes and legend
    plotter.add_axes()
    plotter.add_legend()
    
    # Show the visualization
    plotter.show()

def test_convex_hull(interactive=True):
    """
    Test the convex hull functionality with visualization.
    
    Parameters:
    -----------
    interactive : bool
        Whether to show interactive PyVista visualizations
    """
    # Generate random scattered points
    num_points = 100
    points = []
    for _ in range(num_points):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        z = random.uniform(0, 10)
        points.append([x, y, z])
    
    print(f"Generated {len(points)} random points")
    
    # Convert to numpy arrays for visualization
    points_array = np.array(points)
    
    # Step 1: Visualize the original points
    if interactive:
        print("\nStep 1: Visualizing original points...")
        visualize_with_pyvista(points_array, title="Original Points")
    
    # Compute the convex hull using meshit
    hull_points = meshit.compute_convex_hull(points)
    hull_points_array = np.array(hull_points)
    
    print(f"Convex hull consists of {len(hull_points)} points")
    
    # Use scipy's ConvexHull to get the facets for visualization
    try:
        hull = ConvexHull(hull_points_array)
        
        # Step 2: Visualize the points and convex hull
        if interactive:
            print("\nStep 2: Visualizing points with convex hull...")
            visualize_with_pyvista(points_array, hull_points_array, hull, 
                                  title="Points with Convex Hull")
        
        # Get the triangular facets
        simplices = hull.simplices
        print(f"Convex hull has {len(simplices)} triangular facets")
        
        # Save points and convex hull as VTK files
        save_points_to_vtk(points, "points.vtk")
        save_convex_hull_to_vtk(hull_points_array, hull, "convex_hull.vtk")
        
    except Exception as e:
        print(f"Error creating convex hull visualization: {e}")
        # If scipy's ConvexHull fails, just visualize the points
        if interactive:
            visualize_with_pyvista(points_array, title="Original Points (Hull Failed)")
        
        # Save just the points as VTK file
        save_points_to_vtk(points, "points.vtk")
    
    # Create a matplotlib visualization as well
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
               c='blue', marker='o', alpha=0.5, label='Original Points')
    
    # Plot the convex hull if available
    if 'hull' in locals():
        # Create a Poly3DCollection
        faces = []
        for simplex in simplices:
            faces.append(hull_points_array[simplex])
        
        # Plot the convex hull as a transparent surface
        poly = Poly3DCollection(faces, alpha=0.25, linewidths=1, edgecolor='r')
        poly.set_facecolor('red')
        ax.add_collection3d(poly)
        
        # Plot the hull points
        ax.scatter(hull_points_array[:, 0], hull_points_array[:, 1], hull_points_array[:, 2], 
                   c='red', marker='o', s=50, label='Hull Points')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Convex Hull Visualization')
    ax.legend()
    
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save the figure
    plt.savefig('convex_hull_3d.png', dpi=300)
    print("Enhanced 3D visualization saved as 'convex_hull_3d.png'")
    
    # Create a surface from the points
    surface = meshit.create_surface(points, [], "RandomPoints", "Scattered")
    
    # Calculate the convex hull using the Surface method
    surface.calculate_convex_hull()
    
    # Get the convex hull from the surface
    surface_hull = surface.convex_hull
    
    print(f"Surface convex hull consists of {len(surface_hull)} points")
    
    # Compare the results
    print("\nComparing standalone function and Surface.calculate_convex_hull():")
    if len(hull_points) == len(surface_hull):
        print("Both methods produced the same number of hull points.")
    else:
        print(f"Different number of hull points: {len(hull_points)} vs {len(surface_hull)}")
    
    # Also save the surface convex hull as VTK
    try:
        surface_hull_array = np.array([[p.x, p.y, p.z] for p in surface_hull])
        surface_hull_scipy = ConvexHull(surface_hull_array)
        save_convex_hull_to_vtk(surface_hull_array, surface_hull_scipy, "surface_convex_hull.vtk")
        
        # Step 3: Visualize the surface convex hull
        if interactive:
            print("\nStep 3: Visualizing surface convex hull...")
            visualize_with_pyvista(points_array, surface_hull_array, surface_hull_scipy, 
                                  title="Surface Convex Hull")
    except Exception as e:
        print(f"Error saving surface convex hull to VTK: {e}")
    
    # Step 4: Load and visualize the saved VTK files with PyVista
    if interactive:
        try:
            print("\nStep 4: Loading and visualizing saved VTK files...")
            
            # Load the VTK files
            points_mesh = pv.read("points.vtk")
            hull_mesh = pv.read("convex_hull.vtk")
            surface_hull_mesh = pv.read("surface_convex_hull.vtk")
            
            # Create a plotter with subplots
            plotter = pv.Plotter(shape=(1, 3))
            
            # Plot points
            plotter.subplot(0, 0)
            plotter.add_title("Points")
            plotter.add_mesh(points_mesh, color='blue', point_size=10, render_points_as_spheres=True)
            
            # Plot convex hull
            plotter.subplot(0, 1)
            plotter.add_title("Convex Hull")
            plotter.add_mesh(hull_mesh, color='red', opacity=0.5)
            
            # Plot surface convex hull
            plotter.subplot(0, 2)
            plotter.add_title("Surface Convex Hull")
            plotter.add_mesh(surface_hull_mesh, color='green', opacity=0.5)
            
            # Show the visualization
            plotter.show()
        except Exception as e:
            print(f"Error visualizing VTK files: {e}")
    
    return hull_points, surface_hull

if __name__ == "__main__":
    # Check if running in a non-interactive environment
    if os.environ.get('DISPLAY', '') == '' and not os.name == 'nt':
        print("No display found. Running in non-interactive mode.")
        interactive = False
    else:
        interactive = True
    
    hull_points, surface_hull = test_convex_hull(interactive=interactive) 