import numpy as np
from meshit import Vector3D, Surface
import meshit.extensions

def generate_simple_points():
    """Generate a simple grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, 5):
        for y in np.linspace(0, 10, 5):
            points.append([x, y, 0])
    return points

def print_triangulation_stats(surface):
    """Print basic statistics about the triangulation."""
    if not hasattr(surface, 'triangles') or not surface.triangles:
        print("No triangulation available.")
        return
    
    print(f"Total triangles: {len(surface.triangles)}")
    
    # Calculate triangle areas
    areas = []
    for triangle in surface.triangles:
        vertices = [surface.vertices[i] for i in triangle]
        # Calculate area using cross product
        v1 = np.array([vertices[1].x - vertices[0].x, 
                       vertices[1].y - vertices[0].y, 
                       vertices[1].z - vertices[0].z])
        v2 = np.array([vertices[2].x - vertices[0].x, 
                       vertices[2].y - vertices[0].y, 
                       vertices[2].z - vertices[0].z])
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross)
        areas.append(area)
    
    if areas:
        print(f"Min triangle area: {min(areas):.6f}")
        print(f"Max triangle area: {max(areas):.6f}")
        print(f"Average triangle area: {sum(areas)/len(areas):.6f}")
        print(f"Total surface area: {sum(areas):.6f}")

def test_triangulation(hull_size):
    """Test triangulation with a specific hull size."""
    print(f"\n=== Testing triangulation with hull_size = {hull_size} ===\n")
    
    # Step 1: Create points
    print("Step 1: Creating points")
    raw_points = generate_simple_points()
    print(f"Created {len(raw_points)} points")
    
    # Step 2: Create surface
    print("\nStep 2: Creating surface")
    surface = Surface()
    for point in raw_points:
        surface.add_vertex(Vector3D(point[0], point[1], point[2]))
    print(f"Created surface with {len(surface.vertices)} vertices")
    
    # Step 3: Calculate convex hull
    print("\nStep 3: Calculating convex hull")
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    print(f"Calculated convex hull with {len(surface.convex_hull)} points")
    
    # Step 4: Triangulate
    print("\nStep 4: Triangulating")
    print(f"Using hull size: {hull_size}")
    
    # Direct call to triangulate_with_triangle to avoid any potential issues with the enhanced_triangulate method
    triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
    print(f"Triangulation completed with {len(triangles)} triangles")
    
    # Print statistics
    print("\nTriangulation statistics:")
    print_triangulation_stats(surface)
    
    return surface

if __name__ == "__main__":
    # Test with hull_size = 2.1 (should give approximately 23 triangles)
    test_triangulation(hull_size=2.1) 