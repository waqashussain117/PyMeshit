try:
    import meshit
    print("Successfully imported meshit module")
    print(f"MeshIt version: {meshit.__version__}")
except Exception as e:
    print(f"Error importing meshit: {e}") 