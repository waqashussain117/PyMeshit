import meshit

# Print all available attributes in the meshit module
print("Available attributes in meshit module:")
for attr in dir(meshit):
    if not attr.startswith('__'):
        print(f"- {attr}")

# Try to access the compute_convex_hull function
try:
    print("\nTrying to access compute_convex_hull function...")
    if hasattr(meshit, 'compute_convex_hull'):
        print("compute_convex_hull function is available")
    else:
        print("compute_convex_hull function is NOT available")
except Exception as e:
    print(f"Error: {e}")

# Try to import the _compute_convex_hull function directly
try:
    print("\nTrying to access _compute_convex_hull function...")
    from meshit.core._meshit import compute_convex_hull as _compute_convex_hull
    print("_compute_convex_hull function is available")
except Exception as e:
    print(f"Error: {e}")

# Print the version of the meshit module
print(f"\nMeshIt version: {meshit.__version__}")

# Create a model instance
model = meshit.MeshItModel()

# Print all available attributes and methods
print("Available attributes and methods:")
for attr in dir(model):
    if not attr.startswith('_'):  # Skip private attributes
        print(f"- {attr}")

# Try to create and append a simple surface
vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
triangles = [[0, 1, 2], [0, 2, 3]]

surface = meshit.create_surface(vertices, triangles, "TestSurface", "Planar")
print(f"\nCreated surface with {len(surface.vertices)} vertices")

# Print surface attributes
print("\nSurface attributes:")
for attr in dir(surface):
    if not attr.startswith('_'):
        print(f"- {attr}") 