import numpy as np
import meshit

def generate_grid_points(n=5):
    """Generate a grid of points in the XY plane."""
    points = []
    for x in np.linspace(0, 10, n):
        for y in np.linspace(0, 10, n):
            points.append([x, y, 0])
    return points

def test_triangulation(hull_size):
    """Test triangulation with a specific hull size."""
    print(f"\n=== Testing triangulation with hull_size = {hull_size} ===")
    
    # Step 1: Create points
    raw_points = generate_grid_points()
    print(f"Created {len(raw_points)} points")
    
    # Step 2: Create surface
    surface = meshit.extensions.create_surface_from_points(raw_points)
    print(f"Created surface with {len(surface.vertices)} vertices")
    
    # Step 3: Calculate convex hull
    meshit.extensions.enhanced_calculate_convex_hull(surface)
    print(f"Calculated convex hull with {len(surface.convex_hull)} points")
    
    # Step 4: Triangulate
    triangles = meshit.extensions.triangulate_with_triangle(surface, hull_size=hull_size)
    
    # Print triangle statistics
    print(f"\nTriangulation Statistics:")
    print(f"  Hull size: {hull_size}")
    print(f"  Number of triangles: {len(triangles)}")
    
    # Calculate triangle areas
    areas = []
    for triangle in triangles:
        v1 = surface.vertices[triangle[0]]
        v2 = surface.vertices[triangle[1]]
        v3 = surface.vertices[triangle[2]]
        
        # Calculate area using cross product
        a = np.array([v2.x - v1.x, v2.y - v1.y, v2.z - v1.z])
        b = np.array([v3.x - v1.x, v3.y - v1.y, v3.z - v1.z])
        area = 0.5 * np.linalg.norm(np.cross(a, b))
        areas.append(area)
    
    if areas:
        print(f"  Minimum triangle area: {min(areas):.6f}")
        print(f"  Maximum triangle area: {max(areas):.6f}")
        print(f"  Average triangle area: {sum(areas)/len(areas):.6f}")
    
    return surface, triangles, areas

if __name__ == "__main__":
    # Test with different hull sizes
    hull_sizes = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    results = []
    for hull_size in hull_sizes:
        surface, triangles, areas = test_triangulation(hull_size)
        results.append((hull_size, len(triangles), min(areas) if areas else 0, max(areas) if areas else 0))
        print(f"Completed test with hull_size = {hull_size}, created {len(triangles)} triangles\n")
        print("=" * 50)
    
    # Print summary
    print("\nSummary of Results:")
    print("Hull Size | Number of Triangles | Min Area | Max Area")
    print("-" * 60)
    for hull_size, num_triangles, min_area, max_area in results:
        print(f"{hull_size:8.1f} | {num_triangles:19d} | {min_area:8.6f} | {max_area:8.6f}") 