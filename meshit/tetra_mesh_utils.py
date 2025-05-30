"""
Tetrahedral Mesh Generation Utilities

This module contains the core functionality for generating tetrahedral meshes
using TetGen with proper border handling and material regions.
"""

import numpy as np
import logging
import tetgen
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class TetrahedralMeshGenerator:
    """
    A utility class for generating tetrahedral meshes from surface data.
    
    This class encapsulates all the core functionality for:
    - Collecting and preparing surface data
    - Validating and repairing surface meshes
    - Running TetGen with various strategies
    - Managing material regions and constraints
    - Exporting results
    """
    
    def __init__(self, datasets: List[Dict], selected_surfaces: set, 
                 border_surface_indices: set, unit_surface_indices: set, 
                 fault_surface_indices: set, materials: List[Dict] = None):
        """
        Initialize the tetrahedral mesh generator.
        
        Args:
            datasets: List of surface datasets containing mesh data
            selected_surfaces: Set of selected surface indices
            border_surface_indices: Set of border surface indices
            unit_surface_indices: Set of unit surface indices  
            fault_surface_indices: Set of fault surface indices
            materials: List of material definitions with locations
        """
        self.datasets = datasets
        self.selected_surfaces = selected_surfaces
        self.border_surface_indices = border_surface_indices
        self.unit_surface_indices = unit_surface_indices
        self.fault_surface_indices = fault_surface_indices
        self.materials = materials or []
        self.tetrahedral_mesh = None
        
    def generate_tetrahedral_mesh(self, tetgen_switches: str = "pq1.414aA") -> Optional[Dict]:
        """
        Generate tetrahedral mesh using selected surfaces with C++ MeshIt-style border handling.
        
        Args:
            tetgen_switches: TetGen command line switches
            
        Returns:
            Dictionary containing tetrahedral mesh data or None if failed
        """
        if not self.selected_surfaces:
            logger.warning("No surfaces selected for mesh generation.")
            return None
        
        logger.info("Creating PLC from refined constrained meshes (C++ MeshIt compatible)...")
        
        # Collect surface data with border recognition
        surface_data_dict = self._collect_surface_data()
        
        if not surface_data_dict['vertices'].size:
            logger.error("No surface data available. Complete pre-tetra mesh tab first.")
            return None
        
        # Generate border IDs string like C++ version
        border_ids = self._get_border_ids_string()
        logger.info(f"Border surfaces selected: {border_ids}" if border_ids else "No border surfaces selected")
        
        # Group surfaces by type for proper handling
        unit_surfaces = [i for i in self.selected_surfaces if i in self.unit_surface_indices]
        fault_surfaces = [i for i in self.selected_surfaces if i in self.fault_surface_indices]
        border_surfaces = [i for i in self.selected_surfaces if i in self.border_surface_indices]
        
        logger.info(f"Selected surfaces by type: {len(unit_surfaces)} units, {len(fault_surfaces)} faults, {len(border_surfaces)} borders")
        
        try:
            # Validate and repair surface meshes
            surface_data_dict = self._validate_and_repair_surface_meshes(surface_data_dict)
            
            # Create PyVista mesh from surface data
            vertices = surface_data_dict['vertices']
            triangles = surface_data_dict['triangles']
            surface_markers = surface_data_dict['surface_markers']
            edge_constraints = surface_data_dict.get('edge_constraints', np.array([], dtype=np.int32).reshape(0, 2))
            edge_markers = surface_data_dict.get('edge_markers', np.array([], dtype=np.int32))
            
            logger.info(f"Validating mesh: {len(vertices)} vertices, {len(triangles)} triangles")
            
            # Create triangular faces for PyVista (prepend face size)
            pv_faces = []
            for triangle in triangles:
                pv_faces.extend([3, triangle[0], triangle[1], triangle[2]])
            
            surface_mesh_pv = pv.PolyData(vertices, faces=pv_faces)
            
            # Assign surface markers as cell data
            if len(surface_markers) == len(triangles):
                surface_mesh_pv.cell_data['surface_id'] = surface_markers
            
            logger.info(f"Prepared PLC PyVista mesh: {surface_mesh_pv.n_points} vertices, {surface_mesh_pv.n_faces} facets (triangles), {len(edge_constraints)} PLC edges (segments).")
            
            # Prepare material regions
            region_attributes_list = self._prepare_material_regions(surface_mesh_pv)
            logger.info(f"Prepared {len(region_attributes_list)} material regions")
            
            # Prepare lists for TetGen
            facets_markers_list = surface_markers.tolist() if len(surface_markers) else []
            edges_constraints_list = edge_constraints.tolist() if len(edge_constraints) else []
            edge_markers_list = edge_markers.tolist() if len(edge_markers) else []
            
            # Run TetGen with border-aware strategies
            tetrahedral_grid = self._run_tetgen_with_border_support(
                surface_mesh_pv, 
                facets_markers_list,
                edges_constraints_list, 
                edge_markers_list, 
                region_attributes_list,
                border_ids,
                tetgen_switches
            )
            
            if tetrahedral_grid is not None:
                self.tetrahedral_mesh = tetrahedral_grid
                logger.info("Tetrahedral mesh generation completed successfully")
                return tetrahedral_grid
            else:
                logger.error("TetGen failed to generate mesh")
                return None
        
        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            return None

    def _collect_surface_data(self) -> Dict[str, np.ndarray]:
        """Collect surface data using refined constrained meshes - matching C++ MeshIt behavior."""
        
        # Use C++ MeshIt tolerances
        tolerance = 1e-12  # Match C++ FABS tolerance
        
        all_vertices = []
        all_triangles = []
        surface_markers = []
        edge_constraints = []
        edge_markers = []
        
        vertex_counter = 0
        
        logger.info("Creating PLC from refined constrained meshes (C++ MeshIt compatible)...")
        
        def add_vertex_with_deduplication(vertex, tolerance=1e-12):
            """Add vertex with exact deduplication matching C++ behavior."""
            nonlocal vertex_counter
            
            # Check for existing vertex with strict tolerance (match C++ FABS comparison)
            for existing_idx, existing_vertex in enumerate(all_vertices):
                if (abs(existing_vertex[0] - vertex[0]) < tolerance and 
                    abs(existing_vertex[1] - vertex[1]) < tolerance and 
                    abs(existing_vertex[2] - vertex[2]) < tolerance):
                    return existing_idx
            
            # Add new vertex
            all_vertices.append(list(vertex))
            vertex_id = vertex_counter
            vertex_counter += 1
            return vertex_id
        
        constrained_mesh_used = False
        
        # Process each selected surface using refined constrained mesh data
        for surface_index in self.selected_surfaces:
            if surface_index < len(self.datasets):
                dataset = self.datasets[surface_index]
                
                # Use refined constrained mesh data from pre-tetra mesh tab
                constrained_vertices = dataset.get('constrained_vertices')
                constrained_triangles = dataset.get('constrained_triangles')
                
                if (constrained_vertices is not None and constrained_triangles is not None and 
                    len(constrained_vertices) > 0 and len(constrained_triangles) > 0):
                    
                    logger.info(f"Using REFINED constrained mesh for surface {surface_index}")
                    constrained_mesh_used = True
                    
                    # Add each triangle as a facet (matching C++ Surfaces[s].Ts behavior)
                    vertices_np = np.array(constrained_vertices, dtype=np.float64)
                    
                    # Create local vertex mapping for this surface
                    local_to_global_map = {}
                    for local_idx, vertex_coords in enumerate(vertices_np):
                        global_idx = add_vertex_with_deduplication(vertex_coords, tolerance)
                        local_to_global_map[local_idx] = global_idx
                    
                    # Add all triangles directly (minimal validation - match C++ behavior)
                    valid_triangle_count = 0
                    for triangle_indices in constrained_triangles:
                        if len(triangle_indices) >= 3:
                            try:
                                # Map local indices to global indices
                                global_triangle = [
                                    local_to_global_map[triangle_indices[0]],
                                    local_to_global_map[triangle_indices[1]],
                                    local_to_global_map[triangle_indices[2]]
                                ]
                                
                                # Only check for duplicate vertices (basic validation)
                                if len(set(global_triangle)) == 3:
                                    all_triangles.append(global_triangle)
                                    surface_markers.append(surface_index)  # Match C++ facetmarkerlist[face_number++] = s
                                    valid_triangle_count += 1
                                    
                            except (KeyError, IndexError) as e:
                                logger.warning(f"Skipping invalid triangle in surface {surface_index}: {e}")
                    
                    logger.info(f"Added surface {surface_index}: {len(vertices_np)} vertices, {valid_triangle_count} triangles")
                    
                    # Add intersection constraints if available (match C++ polylines/edge constraints)
                    intersection_constraints = dataset.get('intersection_constraints', [])
                    if intersection_constraints:
                        logger.info(f"Adding {len(intersection_constraints)} intersection constraints for surface {surface_index}")
                        
                        for constraint_line in intersection_constraints:
                            if len(constraint_line) >= 2:
                                # Add constraint points and create edge segments
                                constraint_vertices = []
                                for point_coords in constraint_line:
                                    global_idx = add_vertex_with_deduplication(np.array(point_coords), tolerance)
                                    constraint_vertices.append(global_idx)
                                
                                # Create edge segments from constraint line
                                for i in range(len(constraint_vertices) - 1):
                                    if constraint_vertices[i] != constraint_vertices[i+1]:
                                        edge_constraints.append([constraint_vertices[i], constraint_vertices[i+1]])
                                        edge_markers.append(surface_index + 2)  # Match C++ p+2 (avoid 0,1)
                
                else:
                    logger.error(f"No refined constrained mesh data found for surface {surface_index}")
                    logger.error("Pre-tetra mesh tab should have been completed first!")
        
        # Add global intersection edges if available
        if hasattr(self, 'datasets') and self.datasets:
            for dataset_idx, dataset in enumerate(self.datasets):
                intersections = dataset.get('intersections', [])
                
                for intersection_idx, intersection in enumerate(intersections):
                    intersection_points = intersection.get('points', [])
                    
                    if len(intersection_points) >= 2:
                        # Add intersection line as edge constraints
                        intersection_vertices = []
                        for point in intersection_points:
                            global_idx = add_vertex_with_deduplication(np.array([point[0], point[1], point[2]]), tolerance)
                            intersection_vertices.append(global_idx)
                        
                        # Create edge segments
                        for i in range(len(intersection_vertices) - 1):
                            if intersection_vertices[i] != intersection_vertices[i+1]:
                                edge_constraints.append([intersection_vertices[i], intersection_vertices[i+1]])
                                edge_markers.append(1000 + dataset_idx)  # Global intersection marker
        
        data_source = "REFINED constrained meshes" if constrained_mesh_used else "fallback data"
        logger.info(f"PLC created from {data_source}: {len(all_vertices)} vertices, {len(all_triangles)} triangles, {len(edge_constraints)} edges")
        
        if not constrained_mesh_used:
            logger.error("WARNING: No refined constrained mesh data was used. Please complete pre-tetra mesh tab first!")
        
        return {
            'vertices': np.array(all_vertices, dtype=np.float64),
            'triangles': np.array(all_triangles, dtype=np.int32),
            'surface_markers': np.array(surface_markers, dtype=np.int32),
            'edge_constraints': np.array(edge_constraints, dtype=np.int32) if edge_constraints else np.array([], dtype=np.int32).reshape(0, 2),
            'edge_markers': np.array(edge_markers, dtype=np.int32)
        }

    def _get_border_ids_string(self) -> str:
        """Generate border IDs string like C++ version."""
        if not self.border_surface_indices:
            return ""
        
        # Get intersection of selected surfaces and border surfaces
        selected_borders = self.selected_surfaces.intersection(self.border_surface_indices)
        if not selected_borders:
            return ""
        
        # Sort and create comma-separated string
        border_list = sorted(list(selected_borders))
        return ",".join(str(idx) for idx in border_list)

    def _validate_and_repair_surface_meshes(self, surface_data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Validate and repair surface meshes before tetrahedral generation."""
        logger.info("Validating and repairing surface meshes...")
        
        vertices = surface_data_dict['vertices']
        triangles = surface_data_dict['triangles']
        surface_markers = surface_data_dict['surface_markers']
        
        # Basic validation
        if len(vertices) == 0:
            logger.error("No vertices in surface data")
            return surface_data_dict
            
        if len(triangles) == 0:
            logger.error("No triangles in surface data")
            return surface_data_dict
        
        # Check triangle indices are valid
        max_vertex_index = len(vertices) - 1
        valid_triangles = []
        valid_markers = []
        
        for i, triangle in enumerate(triangles):
            if (len(triangle) >= 3 and 
                all(0 <= idx <= max_vertex_index for idx in triangle[:3]) and
                len(set(triangle[:3])) == 3):  # No duplicate vertices
                valid_triangles.append(triangle[:3])
                if i < len(surface_markers):
                    valid_markers.append(surface_markers[i])
                else:
                    valid_markers.append(0)  # Default marker
        
        logger.info(f"Validation: {len(valid_triangles)}/{len(triangles)} triangles are valid")
        
        surface_data_dict['triangles'] = np.array(valid_triangles, dtype=np.int32)
        surface_data_dict['surface_markers'] = np.array(valid_markers, dtype=np.int32)
        
        return surface_data_dict

    def _prepare_material_regions(self, surface_mesh_pv) -> List[List[float]]:
        """Prepare material regions for TetGen."""
        region_attributes_list = []
        
        if not self.materials:
            # Add default region if no materials defined
            bounds = surface_mesh_pv.bounds
            center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ]
            region_attributes_list.append([center[0], center[1], center[2], 1, -1])  # Default material
            logger.info("Using default material region")
        else:
            for material_idx, material in enumerate(self.materials):
                locations = material.get('locations', [])
                for location in locations:
                    if len(location) >= 3:
                        # Format: [x, y, z, region_attribute, max_volume]
                        region_attributes_list.append([
                            float(location[0]), 
                            float(location[1]), 
                            float(location[2]), 
                            material_idx + 1,  # Material ID (1-based)
                            -1  # No volume constraint
                        ])
                        
            logger.info(f"Added {len(region_attributes_list)} material regions")
        
        return region_attributes_list

    def _run_tetgen_with_border_support(self, surface_mesh_pv, facets_markers_list: List[int], 
                                       edges_constraints_list: List[List[int]], edge_markers_list: List[int], 
                                       region_attributes_list: List[List[float]], border_ids: str, 
                                       plc_switches: str) -> Optional[pv.UnstructuredGrid]:
        """Run TetGen with proper border handling like C++ version."""
        
        # Create TetGen object with border awareness
        tet = tetgen.TetGen(surface_mesh_pv)
        
        # Add border-specific constraints
        if border_ids:
            logger.info(f"Processing border constraints for border IDs: {border_ids}")
            # Borders get special treatment - they define domain boundaries
            tet.make_manifold(verbose=False)
        
        # Progressive strategy approach with border consideration
        strategies = [
            ('A', 'Basic Delaunay (no boundary recovery) - BEST for constrained meshes'),
            ('pA', 'Basic with PLC - Good for borders'),
            ('pq1.2A', 'Quality with PLC - Balanced approach'),
            (plc_switches, 'User-specified switches'),
            ('pq1.414aAY', 'Final fallback with all options')
        ]
        
        for switches, description in strategies:
            try:
                logger.info(f"Trying TetGen strategy: '{switches}' ({description})")
                
                # Configure tetgen based on strategy
                if border_ids and 'p' not in switches:
                    # For borders, ensure PLC mode is used
                    switches = 'p' + switches
                    logger.info(f"Added PLC mode for borders: '{switches}'")
                
                grid = tet.tetrahedralize(
                    switches=switches,
                    facetmarkerlist=facets_markers_list,
                    edgelist=edges_constraints_list if len(edges_constraints_list) else None,
                    edgemarkerlist=edge_markers_list if len(edge_markers_list) else None,
                    regionlist=region_attributes_list if len(region_attributes_list) else None
                )
                
                if grid and grid.n_cells > 0:
                    logger.info(f"✓ Strategy '{switches}' succeeded!")
                    logger.info(f"  Generated {grid.n_cells} tetrahedra in minimal time")
                    logger.info(f"  Vertices: {grid.n_points}")
                    logger.info(f"  Used edge constraints: {len(edges_constraints_list) > 0}")
                    
                    if border_ids:
                        logger.info(f"  Border handling: Applied for IDs {border_ids}")
                    
                    return grid
                else:
                    logger.warning(f"Strategy '{switches}' produced no tetrahedra")
                    
            except Exception as e:
                logger.warning(f"Strategy '{switches}' failed: {str(e)}")
                continue
        
        logger.error("All TetGen strategies failed")
        return None

    def export_mesh(self, file_path: str, mesh_data: Optional[Dict] = None) -> bool:
        """
        Export tetrahedral mesh to various formats.
        
        Args:
            file_path: Output file path
            mesh_data: Mesh data to export (uses self.tetrahedral_mesh if None)
            
        Returns:
            True if export successful, False otherwise
        """
        if mesh_data is None:
            mesh_data = self.tetrahedral_mesh
            
        if not mesh_data:
            logger.error("No tetrahedral mesh to export")
            return False
        
        try:
            if isinstance(mesh_data, pv.UnstructuredGrid):
                # Direct PyVista grid export
                mesh_data.save(file_path)
                logger.info(f"Tetrahedral mesh exported to: {file_path}")
                return True
            elif isinstance(mesh_data, dict):
                # Manual export using mesh data
                return self._export_manual(file_path, mesh_data)
            else:
                logger.error(f"Unsupported mesh data type: {type(mesh_data)}")
                return False
                
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False

    def _export_manual(self, file_path: str, mesh_data: Dict) -> bool:
        """Manual export when PyVista grid is not available."""
        vertices = mesh_data.get('vertices')
        tetrahedra = mesh_data.get('tetrahedra')
        
        if vertices is None or tetrahedra is None:
            logger.error("Missing vertices or tetrahedra data")
            return False
        
        try:
            if file_path.endswith('.ply'):
                return self._export_ply(file_path, vertices, tetrahedra)
            else:
                # Default to simple text format
                with open(file_path, 'w') as f:
                    f.write(f"# Tetrahedral Mesh\n")
                    f.write(f"# Vertices: {len(vertices)}\n")
                    f.write(f"# Tetrahedra: {len(tetrahedra)}\n\n")
                    
                    f.write("VERTICES\n")
                    for v in vertices:
                        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    
                    f.write("\nTETRAHEDRA\n")
                    for t in tetrahedra:
                        f.write(f"{t[0]} {t[1]} {t[2]} {t[3]}\n")
                
                logger.info(f"Manual export completed: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Manual export failed: {str(e)}")
            return False

    def _export_ply(self, file_path: str, vertices: np.ndarray, tetrahedra: np.ndarray) -> bool:
        """Export tetrahedral mesh to PLY format."""
        try:
            with open(file_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write(f"element face {len(tetrahedra)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                # Write vertices
                for v in vertices:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Write tetrahedra as 4-vertex faces
                for t in tetrahedra:
                    f.write(f"4 {t[0]} {t[1]} {t[2]} {t[3]}\n")
            
            logger.info(f"PLY export completed: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"PLY export failed: {str(e)}")
            return False

    def get_mesh_statistics(self, mesh_data: Optional[Dict] = None) -> Dict[str, Union[int, float]]:
        """
        Calculate mesh statistics.
        
        Args:
            mesh_data: Mesh data to analyze (uses self.tetrahedral_mesh if None)
            
        Returns:
            Dictionary with mesh statistics
        """
        if mesh_data is None:
            mesh_data = self.tetrahedral_mesh
            
        if not mesh_data:
            return {}
        
        stats = {}
        
        if isinstance(mesh_data, pv.UnstructuredGrid):
            stats['n_vertices'] = mesh_data.n_points
            stats['n_tetrahedra'] = mesh_data.n_cells
            stats['volume'] = float(mesh_data.volume) if hasattr(mesh_data, 'volume') else 0.0
            
        elif isinstance(mesh_data, dict):
            vertices = mesh_data.get('vertices')
            tetrahedra = mesh_data.get('tetrahedra')
            
            if vertices is not None:
                stats['n_vertices'] = len(vertices)
            if tetrahedra is not None:
                stats['n_tetrahedra'] = len(tetrahedra)
                
            # Calculate volume if data is available
            if vertices is not None and tetrahedra is not None:
                stats['volume'] = self._calculate_mesh_volume(vertices, tetrahedra)
        
        return stats

    def _calculate_mesh_volume(self, vertices: np.ndarray, tetrahedra: np.ndarray) -> float:
        """Calculate total volume of tetrahedral mesh."""
        total_volume = 0.0
        
        for tet in tetrahedra:
            if len(tet) >= 4:
                # Get tetrahedron vertices
                v0 = vertices[tet[0]]
                v1 = vertices[tet[1]]
                v2 = vertices[tet[2]]
                v3 = vertices[tet[3]]
                
                # Calculate volume using determinant formula
                # V = |det([v1-v0, v2-v0, v3-v0])| / 6
                matrix = np.array([
                    v1 - v0,
                    v2 - v0,
                    v3 - v0
                ])
                
                volume = abs(np.linalg.det(matrix)) / 6.0
                total_volume += volume
        
        return total_volume


def create_tetrahedral_mesh(datasets: List[Dict], selected_surfaces: set, 
                           border_surface_indices: set, unit_surface_indices: set, 
                           fault_surface_indices: set, materials: List[Dict] = None,
                           tetgen_switches: str = "pq1.414aA") -> Optional[Dict]:
    """
    Convenience function to create a tetrahedral mesh.
    
    Args:
        datasets: List of surface datasets containing mesh data
        selected_surfaces: Set of selected surface indices
        border_surface_indices: Set of border surface indices
        unit_surface_indices: Set of unit surface indices  
        fault_surface_indices: Set of fault surface indices
        materials: List of material definitions with locations
        tetgen_switches: TetGen command line switches
        
    Returns:
        Tetrahedral mesh data or None if failed
    """
    generator = TetrahedralMeshGenerator(
        datasets=datasets,
        selected_surfaces=selected_surfaces,
        border_surface_indices=border_surface_indices,
        unit_surface_indices=unit_surface_indices,
        fault_surface_indices=fault_surface_indices,
        materials=materials
    )
    
    return generator.generate_tetrahedral_mesh(tetgen_switches) 