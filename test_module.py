import meshit

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