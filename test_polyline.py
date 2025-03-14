import meshit
import numpy as np

def test_polyline_creation():
    """Test creating a polyline and adding vertices to it."""
    print("Testing polyline creation...")
    
    # Create some points
    points = []
    for i in range(5):
        points.append([i, i*2, i*3])
    
    print(f"Created {len(points)} points")
    
    # Create a polyline using the create_polyline function
    polyline = meshit.create_polyline(points, "TestPolyline")
    polyline.size = 0.5
    
    print(f"After creation, polyline has {len(polyline.vertices)} vertices")
    
    # Print the vertices
    if polyline.vertices:
        print("\nPolyline vertices:")
        for i, vertex in enumerate(polyline.vertices):
            print(f"  Vertex {i}: ({vertex.x}, {vertex.y}, {vertex.z})")
    
    # Print bounds
    if hasattr(polyline, 'bounds'):
        print(f"\nPolyline bounds: Min({polyline.bounds[0].x}, {polyline.bounds[0].y}, {polyline.bounds[0].z}), "
              f"Max({polyline.bounds[1].x}, {polyline.bounds[1].y}, {polyline.bounds[1].z})")
    
    # Calculate segments
    print("\nPerforming coarse segmentation...")
    polyline.calculate_segments(False)
    
    print(f"After segmentation, polyline has {len(polyline.segments)} segments")
    
    # Print the segments
    if polyline.segments:
        print("\nPolyline segments:")
        for i, segment in enumerate(polyline.segments):
            print(f"  Segment {i}: {segment}")
    
    return polyline

if __name__ == "__main__":
    polyline = test_polyline_creation() 