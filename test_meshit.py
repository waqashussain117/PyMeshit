import meshit
import math
import os
import time

def create_shape(shape_type, **params):
    """Generate points for different parametric shapes"""
    points = []
    
    if shape_type == "rectangle":
        width = params.get("width", 10.0)
        height = params.get("height", 5.0)
        z = params.get("z", 0.0)
        points = [
            [0.0, 0.0, z],
            [width, 0.0, z],
            [width, height, z],
            [0.0, height, z],
            [0.0, 0.0, z]  # Close the loop
        ]
    
    elif shape_type == "circle":
        cx = params.get("cx", 0.0)
        cy = params.get("cy", 0.0)
        r = params.get("radius", 5.0)
        z = params.get("z", 0.0)
        segments = params.get("segments", 36)
        
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append([x, y, z])
    
    elif shape_type == "hexagon":
        cx = params.get("cx", 0.0)
        cy = params.get("cy", 0.0)
        r = params.get("radius", 5.0)
        z = params.get("z", 0.0)
        
        for i in range(7):  # 6 points + 1 to close the loop
            angle = 2 * math.pi * i / 6
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append([x, y, z])
            
    elif shape_type == "donut":
        cx = params.get("cx", 0.0)
        cy = params.get("cy", 0.0)
        r_outer = params.get("outer_radius", 5.0)
        r_inner = params.get("inner_radius", 2.5)
        z = params.get("z", 0.0)
        segments = params.get("segments", 36)
        
        # Outer circle
        outer_points = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = cx + r_outer * math.cos(angle)
            y = cy + r_outer * math.sin(angle)
            outer_points.append([x, y, z])
        outer_points.append(outer_points[0].copy())  # Close the loop
        
        # Inner circle (going counter-clockwise)
        inner_points = []
        for i in range(segments, 0, -1):
            angle = 2 * math.pi * i / segments
            x = cx + r_inner * math.cos(angle)
            y = cy + r_inner * math.sin(angle)
            inner_points.append([x, y, z])
        inner_points.append(inner_points[0].copy())  # Close the loop
        
        # Return both (for separate processing)
        return [outer_points, inner_points]
        
    return points

def process_surfaces():
    """Create and process multiple surfaces with MeshIt"""
    print("Starting surface processing with MeshIt")
    
    # Initialize the model
    model = meshit.MeshItModel()
    model.set_mesh_quality(1.5)  # Higher quality
    model.set_mesh_algorithm("delaunay")  # Use Delaunay algorithm
    
    # Create different shapes
    print("\nGenerating shapes...")
    
    # Rectangle
    rect = create_shape("rectangle", width=10.0, height=7.0, z=0.0)
    model.add_polyline(rect)
    print(f"Added rectangle with {len(rect)} points")
    
    # Circle
    circle = create_shape("circle", cx=15.0, cy=5.0, radius=3.5, z=0.0)
    model.add_polyline(circle)
    print(f"Added circle with {len(circle)} points")
    
    # Hexagon
    hexagon = create_shape("hexagon", cx=5.0, cy=15.0, radius=4.0, z=0.0)
    model.add_polyline(hexagon)
    print(f"Added hexagon with {len(hexagon)} points")
    
    # Donut (needs special processing - outer and inner boundaries)
    donut_shapes = create_shape("donut", cx=15.0, cy=15.0, outer_radius=5.0, inner_radius=2.0, z=0.0)
    model.add_polyline(donut_shapes[0])  # Outer circle
    model.add_polyline(donut_shapes[1])  # Inner circle
    print(f"Added donut with outer boundary ({len(donut_shapes[0])} points) and inner boundary ({len(donut_shapes[1])} points)")
    
    # Process the mesh
    print("\nPre-meshing surfaces...")
    start_time = time.time()
    model.pre_mesh()
    
    print("Generating mesh...")
    model.enable_constraints(True)  # Enable boundary constraints
    model.mesh()
    mesh_time = time.time() - start_time
    print(f"Mesh generation completed in {mesh_time:.3f} seconds")
    
    # Export the result
    output_file = "surface_mesh_results.vtu"
    print(f"\nExporting mesh to {output_file}...")
    model.export_vtu(output_file)
    
    # Verify the file was created
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"Success! Exported VTU file ({file_size:.2f} KB)")
        print(f"File path: {os.path.abspath(output_file)}")
        print("\nYou can view this file in ParaView or other VTK-compatible viewers")
    else:
        print("Error: Failed to create output file")
    
    return output_file

if __name__ == "__main__":
    try:
        output_file = process_surfaces()
        print("\nSurface processing completed successfully!")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")