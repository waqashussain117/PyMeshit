import meshit
import numpy as np

def progress_callback(message):
    """Callback function to print progress messages"""
    print(message)

def test_premesh():
    # Import necessary modules
    import meshit
    
    # Create a MeshItModel instance
    model = meshit.MeshItModel()

    # Create a simple surface (a square in the XY plane)
    square_vertices = [
        [0.0, 0.0, 0.0],  # vertex 0
        [1.0, 0.0, 0.0],  # vertex 1
        [1.0, 1.0, 0.0],  # vertex 2
        [0.0, 1.0, 0.0],  # vertex 3
    ]
    square_triangles = [
        [0, 1, 2],  # triangle 1
        [0, 2, 3],  # triangle 2
    ]

    # Create another surface (a vertical square that clearly intersects the first one)
    vertical_vertices = [
        [0.5, 0.5, -0.5],   # vertex 0
        [0.5, 0.5, 0.5],    # vertex 1
        [0.5, 1.5, 0.5],    # vertex 2
        [0.5, 1.5, -0.5],   # vertex 3
    ]
    vertical_triangles = [
        [0, 1, 2],  # triangle 1
        [0, 2, 3],  # triangle 2
    ]

    # Create a polyline (a vertical line that intersects the first surface)
    line_vertices = [
        [0.25, 0.25, -0.5],  # start point
        [0.25, 0.25, 0.5],   # end point
    ]

    print("\nCreating geometries...")
    
    # Create surfaces and polyline
    surface1 = meshit.create_surface(square_vertices, square_triangles, "Square", "Planar")
    print(f"Surface 1 created with {len(surface1.vertices)} vertices and {len(surface1.triangles)} triangles")
    
    surface2 = meshit.create_surface(vertical_vertices, vertical_triangles, "VerticalSquare", "Planar")
    print(f"Surface 2 created with {len(surface2.vertices)} vertices and {len(surface2.triangles)} triangles")
    
    polyline = meshit.create_polyline(line_vertices, "Intersecting_Line")
    print(f"Polyline created with {len(polyline.vertices)} vertices")

    # Add geometries to the model using helper functions
    print("\nAdding geometries to the model...")
    meshit.add_surface_to_model(model, surface1)
    meshit.add_surface_to_model(model, surface2)
    meshit.add_polyline_to_model(model, polyline)
    
    # Verify that geometries were added correctly
    print("\nModel state after adding geometries:")
    print(f"- Surfaces: {len(model.surfaces)}")
    for i, surface in enumerate(model.surfaces):
        print(f"  Surface {i+1}: {surface.name} with {len(surface.vertices)} vertices and {len(surface.triangles)} triangles")
    print(f"- Polylines: {len(model.model_polylines)}")
    for i, polyline in enumerate(model.model_polylines):
        print(f"  Polyline {i+1}: {polyline.name} with {len(polyline.vertices)} vertices")

    # Run pre_mesh_job with progress callback
    print("\nStarting pre_mesh_job...")
    try:
        model.pre_mesh_job(progress_callback)
        print("\nPre-mesh job completed successfully!")
        
        # Print some results
        print("\nResults:")
        
        # Get intersections using the helper function
        intersections = meshit.get_intersections(model)
        print(f"Number of intersections found: {len(intersections)}")
        for i, intersection in enumerate(intersections):
            print(f"\nIntersection {i + 1}:")
            print(f"- Between {'polyline' if intersection.is_polyline_mesh else 'surface'} {intersection.id1} and surface {intersection.id2}")
            print(f"- Number of intersection points: {len(intersection.points)}")
            if len(intersection.points) > 0:
                print("- Intersection points:")
                for j, point in enumerate(intersection.points):
                    print(f"  Point {j+1}: ({point.x:.3f}, {point.y:.3f}, {point.z:.3f})")
        
        # Get triple points using the helper function
        triple_points = meshit.get_triple_points(model)
        print(f"\nNumber of triple points found: {len(triple_points)}")
        for i, tp in enumerate(triple_points):
            print(f"\nTriple point {i + 1}:")
            print(f"- Position: ({tp.point.x:.3f}, {tp.point.y:.3f}, {tp.point.z:.3f})")
            print(f"- Involved intersections: {tp.intersection_ids}")

    except Exception as e:
        print(f"Error during pre-mesh job: {str(e)}")

if __name__ == "__main__":
    test_premesh() 