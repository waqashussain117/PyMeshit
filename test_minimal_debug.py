import numpy as np
import meshit
import sys

def generate_simple_points():
    """Generate a simple grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, 5):
        for y in np.linspace(0, 10, 5):
            points.append([x, y, 0])
    return points

def main():
    print("=== Minimal Triangulation Test ===")
    
    # Set hull size
    hull_size = 2.1
    print(f"Using hull_size = {hull_size}")
    
    # Step 1: Create points
    print("\nStep 1: Creating points")
    raw_points = generate_simple_points()
    print(f"Created {len(raw_points)} points")
    
    # Step 2: Create surface
    print("\nStep 2: Creating surface")
    surface = meshit.extensions.create_surface_from_points(raw_points)
    print(f"Created surface with {len(surface.vertices)} vertices")
    
    # Step 3: Calculate convex hull
    print("\nStep 3: Calculating convex hull")
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    print(f"Calculated convex hull with {len(surface.convex_hull)} points")
    
    # Step 4: Triangulate
    print("\nStep 4: Triangulating")
    try:
        print(f"Starting triangulation with hull_size = {hull_size}")
        triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
        print(f"Triangulation completed with {len(triangles)} triangles")
        
        # Print triangle details
        if triangles:
            print("\nTriangle details:")
            for i, triangle in enumerate(triangles[:5]):  # Print first 5 triangles
                print(f"Triangle {i}: {triangle}")
            if len(triangles) > 5:
                print(f"... and {len(triangles) - 5} more triangles")
    except Exception as e:
        print(f"Error during triangulation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed")

if __name__ == "__main__":
    main() 