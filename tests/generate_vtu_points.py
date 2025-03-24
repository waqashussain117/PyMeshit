#!/usr/bin/env python
"""
Generate the exact 25 random points used in our triangulation tests
and save them in VTU format for manual testing in MeshIt.
"""

import os
import numpy as np
from scipy.spatial import ConvexHull
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def generate_points(num_points=25, seed=42):
    """Generate the exact same random points used in our triangulation tests"""
    np.random.seed(seed)
    points = np.random.uniform(-1, 1, (num_points, 2))
    return points

def compute_convex_hull(points):
    """Compute the convex hull of points"""
    hull = ConvexHull(points)
    hull_vertices = hull.vertices
    hull_points = points[hull_vertices]
    return hull_points

def write_vtu_file(points, output_file):
    """
    Write points to a VTU file format that MeshIt can read.
    Since we're only interested in the points, we'll create a minimal
    VTU file with just the point data.
    """
    # Convert 2D points to 3D by adding z=0
    points_3d = np.column_stack((points, np.zeros(len(points))))
    
    # Create the VTK/VTU XML structure
    vtk = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
    grid = ET.SubElement(vtk, "UnstructuredGrid")
    piece = ET.SubElement(grid, "Piece", NumberOfPoints=str(len(points)), NumberOfCells="0")
    
    # Add point data
    point_data = ET.SubElement(piece, "PointData")
    
    # Add cell data (empty)
    cell_data = ET.SubElement(piece, "CellData")
    
    # Add points
    points_elem = ET.SubElement(piece, "Points")
    data_array = ET.SubElement(points_elem, "DataArray", type="Float32", NumberOfComponents="3", format="ascii")
    data_array.text = " ".join([f"{x} {y} {z}" for x, y, z in points_3d])
    
    # Add cells (empty)
    cells = ET.SubElement(piece, "Cells")
    
    # Connectivity (empty since we have no cells)
    conn = ET.SubElement(cells, "DataArray", type="Int32", Name="connectivity", format="ascii")
    conn.text = ""
    
    # Offsets (empty since we have no cells)
    offsets = ET.SubElement(cells, "DataArray", type="Int32", Name="offsets", format="ascii")
    offsets.text = ""
    
    # Types (empty since we have no cells)
    types = ET.SubElement(cells, "DataArray", type="UInt8", Name="types", format="ascii")
    types.text = ""
    
    # Convert to pretty-printed XML
    rough_string = ET.tostring(vtk, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(pretty_xml)

def main():
    # Create output directory
    output_dir = "vtu_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate points (same as in our triangulation tests)
    points = generate_points(25)
    
    # Compute convex hull
    hull_points = compute_convex_hull(points)
    
    # Write points to VTU file
    write_vtu_file(points, os.path.join(output_dir, "test_points.vtu"))
    
    # Write hull points to VTU file
    write_vtu_file(hull_points, os.path.join(output_dir, "hull_points.vtu"))
    
    # Write points as text file for easy reference
    with open(os.path.join(output_dir, "points.txt"), 'w') as f:
        f.write("# Format: x y\n")
        for i, (x, y) in enumerate(points):
            f.write(f"Point {i+1}: {x:.6f} {y:.6f}\n")
    
    # Write hull points as text file
    with open(os.path.join(output_dir, "hull_points.txt"), 'w') as f:
        f.write("# Format: x y\n")
        for i, (x, y) in enumerate(hull_points):
            f.write(f"Hull Point {i+1}: {x:.6f} {y:.6f}\n")
    
    print(f"Generated VTU files in {output_dir}/:")
    print(f"  - test_points.vtu: Contains all 25 points")
    print(f"  - hull_points.vtu: Contains only the convex hull points")
    print(f"  - points.txt: Text file with all point coordinates")
    print(f"  - hull_points.txt: Text file with hull point coordinates")

if __name__ == "__main__":
    main() 