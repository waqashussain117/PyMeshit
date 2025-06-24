"""
Tetrahedral Mesh Generation Utilities

This module contains the core functionality for generating tetrahedral meshes
using TetGen with C++ MeshIt compatible approach (separate surfaces).
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
    
    This class follows the C++ MeshIt approach exactly:
    - Keeps surfaces separate with individual triangulations
    - Uses precise vertex matching to find shared vertices
    - Passes separate surface data + intersection constraints to TetGen
    - No geometric merging of surfaces
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
        Generate tetrahedral mesh using C++ MeshIt approach.
        
        Args:
            tetgen_switches: TetGen command line switches
            
        Returns:
            Dictionary containing tetrahedral mesh data or None if failed
        """
        if not self.selected_surfaces:
            logger.warning("No surfaces selected for mesh generation.")
            return None
        
        logger.info("Creating PLC using C++ MeshIt approach (separate surfaces)...")
        
        try:
            # Step 1: Collect surface data without merging
            surface_data = self._collect_separate_surface_data()
            
            if not surface_data['vertices'].size:
                logger.error("No surface data available. Complete pre-tetra mesh tab first.")
                return None
            
            # Step 2: Create PyVista mesh from separate surface data
            surface_mesh_pv = self._create_pyvista_mesh(surface_data)
            
            if surface_mesh_pv is None:
                logger.error("Failed to create PyVista mesh")
                return None
            
            logger.info(f"Created PLC: {surface_mesh_pv.n_points} vertices, {surface_mesh_pv.n_faces} facets, {len(surface_data['edge_constraints'])} constraints")
            
            # Step 3: Prepare material regions
            region_attributes_list = self._prepare_material_regions(surface_mesh_pv)
            logger.info(f"Prepared {len(region_attributes_list)} material regions")
            
            # Step 4: Generate border IDs string
            border_ids = self._get_border_ids_string()
            logger.info(f"Border surfaces selected: {border_ids}" if border_ids else "No border surfaces selected")
            
            # Step 5: Run TetGen with C++ compatible approach
            tetrahedral_grid = self._run_tetgen_cpp_style(
                surface_mesh_pv, 
                surface_data['surface_markers'],
                surface_data['edge_constraints'], 
                surface_data['edge_markers'], 
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

    def _collect_separate_surface_data(self) -> Dict[str, np.ndarray]:
        """
        Collect surface data using C++ approach: separate surfaces with precise vertex matching.
        
        This replicates the C++ calculate_tets function exactly:
        1. Process each surface individually 
        2. Find shared vertices using 1e-12 tolerance (like C++)
        3. Keep surface triangulations separate with proper markers
        4. Add intersection polylines as edge constraints
        """
        logger.info("Collecting surface data using C++ approach...")
        
        # Global vertex list with precise matching (like C++ pointlist)
        global_vertices = []  # [x, y, z, x, y, z, ...]
        vertex_tet_ids = []   # Global vertex indices
        tolerance = 1e-12     # C++ uses 1e-12 for vertex matching
        
        # Step 1: Process surface vertices (like C++ Surfaces loop)
        surface_vertex_maps = {}  # surface_idx -> {local_idx: global_idx}
        
        for s_idx in self.selected_surfaces:
            logger.info(f"Processing surface {s_idx}...")
            ds = self.datasets[s_idx]
            
            # Get constrained vertices (from pre-tetra mesh tab)
            vertices = ds.get("constrained_vertices")
            if vertices is None or len(vertices) == 0:
                vertices = ds.get("triangulated_vertices", [])
            
            if len(vertices) == 0:
                logger.warning(f"Surface {s_idx} has no vertices. Skipping.")
                continue
                
            surface_vertex_maps[s_idx] = {}
            
            # Process each vertex in this surface
            for local_idx, vertex in enumerate(vertices):
                # Find existing vertex with precise matching (like C++)
                found_global_idx = None
                x, y, z = float(vertex[0]), float(vertex[1]), float(vertex[2])
                
                # Search existing vertices for match
                for global_idx in range(len(vertex_tet_ids)):
                    existing_x = global_vertices[global_idx * 3]
                    existing_y = global_vertices[global_idx * 3 + 1] 
                    existing_z = global_vertices[global_idx * 3 + 2]
                    
                    if (abs(x - existing_x) < tolerance and 
                        abs(y - existing_y) < tolerance and 
                        abs(z - existing_z) < tolerance):
                        found_global_idx = global_idx
                        break
                
                if found_global_idx is None:
                    # Add new vertex
                    global_vertices.extend([x, y, z])
                    global_idx = len(vertex_tet_ids)
                    vertex_tet_ids.append(global_idx)
                else:
                    # Use existing vertex
                    global_idx = found_global_idx
                    
                surface_vertex_maps[s_idx][local_idx] = global_idx
        
        logger.info(f"Global vertex processing complete: {len(vertex_tet_ids)} unique vertices")
        
        # Step 2: Process surface triangles (like C++ facet creation)
        all_triangles = []
        surface_markers = []
        
        for s_idx in self.selected_surfaces:
            ds = self.datasets[s_idx]
            triangles = ds.get("constrained_triangles")
            if triangles is None or len(triangles) == 0:
                triangles = ds.get("triangulated_triangles", [])
                
            if len(triangles) == 0 or s_idx not in surface_vertex_maps:
                continue
                
            vertex_map = surface_vertex_maps[s_idx]
            
            # Convert triangles to global indices
            for triangle in triangles:
                if len(triangle) >= 3:
                    try:
                        global_tri = [
                            vertex_map[int(triangle[0])],
                            vertex_map[int(triangle[1])], 
                            vertex_map[int(triangle[2])]
                        ]
                        # Check for degenerate triangles
                        if len(set(global_tri)) == 3:
                            all_triangles.append(global_tri)
                            surface_markers.append(s_idx)  # Use surface index as marker
                    except (KeyError, ValueError, IndexError):
                        continue
        
        logger.info(f"Triangle processing complete: {len(all_triangles)} triangles from {len(self.selected_surfaces)} surfaces")
        
        # Step 3: Process intersection polylines as edge constraints (like C++ Polylines)
        edge_constraints = []
        edge_markers = []
        
        for s_idx in self.selected_surfaces:
            ds = self.datasets[s_idx]
            
            # Get intersection constraints from stored_constraints
            for constraint in ds.get('stored_constraints', []):
                if constraint.get('type') != 'intersection_line':
                    continue
                    
                points = constraint.get('points', [])
                if len(points) < 2:
                    continue
                
                # Find global indices for intersection points
                global_indices = []
                for point in points:
                    x, y, z = float(point[0]), float(point[1]), float(point[2])
                    
                    # Find matching global vertex
                    found_idx = None
                    for global_idx in range(len(vertex_tet_ids)):
                        existing_x = global_vertices[global_idx * 3]
                        existing_y = global_vertices[global_idx * 3 + 1]
                        existing_z = global_vertices[global_idx * 3 + 2]
                        
                        if (abs(x - existing_x) < tolerance and 
                            abs(y - existing_y) < tolerance and 
                            abs(z - existing_z) < tolerance):
                            found_idx = global_idx
                            break
                    
                    if found_idx is None:
                        # Add new vertex for intersection point
                        global_vertices.extend([x, y, z])
                        found_idx = len(vertex_tet_ids)
                        vertex_tet_ids.append(found_idx)
                    
                    global_indices.append(found_idx)
                
                # Create edge constraints for this polyline
                for i in range(len(global_indices) - 1):
                    edge_constraints.append([global_indices[i], global_indices[i + 1]])
                    edge_markers.append(-1)  # Generic marker for intersections
        
        logger.info(f"Edge constraint processing complete: {len(edge_constraints)} edge constraints")
        
        # Convert to numpy arrays 
        vertices_array = np.array(global_vertices, dtype=np.float64).reshape(-1, 3)
        triangles_array = np.array(all_triangles, dtype=np.int32)
        markers_array = np.array(surface_markers, dtype=np.int32)
        constraints_array = np.array(edge_constraints, dtype=np.int32) if edge_constraints else np.array([], dtype=np.int32).reshape(0, 2)
        edge_markers_array = np.array(edge_markers, dtype=np.int32)
        
        return {
            'vertices': vertices_array,
            'triangles': triangles_array, 
            'surface_markers': markers_array,
            'edge_constraints': constraints_array,
            'edge_markers': edge_markers_array
        }

    def _create_pyvista_mesh(self, surface_data: Dict) -> Optional[pv.PolyData]:
        """Create PyVista mesh from separate surface data."""
        vertices = surface_data['vertices']
        triangles = surface_data['triangles']
        surface_markers = surface_data['surface_markers']
        
        if len(vertices) == 0 or len(triangles) == 0:
            logger.error("No vertices or triangles to create mesh")
            return None
            
        # Create PyVista faces format
        pv_faces = []
        for triangle in triangles:
            pv_faces.extend([3, triangle[0], triangle[1], triangle[2]])
        
        try:
            mesh = pv.PolyData(vertices, faces=pv_faces)
            
            # Add surface markers as cell data
            if len(surface_markers) == len(triangles):
                mesh.cell_data['surface_id'] = surface_markers
                
            return mesh
        except Exception as e:
            logger.error(f"Failed to create PyVista mesh: {e}")
            return None

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

    def _prepare_material_regions(self, surface_mesh_pv) -> List[List[float]]:
        """Prepare material regions for TetGen (C++ compatible)."""
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
                # Use the material's attribute if specified, otherwise use index + 1
                material_attribute = material.get('attribute', material_idx + 1)
                
                for location in locations:
                    if len(location) >= 3:
                        # Format: [x, y, z, region_attribute, max_volume]
                        region_attributes_list.append([
                            float(location[0]), 
                            float(location[1]), 
                            float(location[2]), 
                            int(material_attribute),  # Use specified attribute
                            -1  # No volume constraint
                        ])
                        
            logger.info(f"Added {len(region_attributes_list)} material regions from {len(self.materials)} materials")
        
        return region_attributes_list

    def _run_tetgen_cpp_style(
            self,
            surface_mesh_pv,                 # PyVista PolyData (PLC facets)
            surface_markers: np.ndarray,     # Surface markers for each facet
            edge_constraints: np.ndarray,    # Edge constraints array 
            edge_markers: np.ndarray,        # Edge markers array
            region_attributes_list: list,
            border_ids: str,
            plc_switches: str,
    ) -> Optional[pv.UnstructuredGrid]:
        """
        Execute TetGen using C++ compatible approach with proper surface markers.
        """
        import tetgen
        import logging

        logger = logging.getLogger(__name__)

        # Clean the input mesh
        try:
            surface_mesh_pv.clean(inplace=True)
        except Exception:
            pass

        def _create_tetgen_instance():
            """Helper to create and configure a new TetGen instance with proper surface markers."""
            tet = tetgen.TetGen(surface_mesh_pv)
            
            # CRITICAL: Set surface markers properly (like C++ facetmarkerlist)
            if len(surface_markers) > 0:
                # Convert surface markers to list and ensure they match triangle count
                marker_list = surface_markers.tolist()
                if len(marker_list) == surface_mesh_pv.n_faces:
                    tet.facet_markers = marker_list
                    logger.info(f"Applied {len(marker_list)} surface markers to TetGen")
                else:
                    logger.warning(f"Surface marker count ({len(marker_list)}) does not match face count ({surface_mesh_pv.n_faces})")
            
            # Set edge constraints (like C++ edgelist)
            if len(edge_constraints) > 0:
                edge_list = edge_constraints.tolist()
                tet.edge_list = edge_list
                logger.info(f"Applied {len(edge_list)} edge constraints to TetGen")
                
                # Set edge markers if available
                if len(edge_markers) > 0:
                    tet.edge_markers = edge_markers.tolist()
            
            # Add material regions (like C++ regionlist)
            for reg in region_attributes_list:
                try:
                    tet.add_region(reg[:3], attribute=int(reg[3]), max_volume=float(reg[4]))
                except AttributeError:
                    # Fallback for older TetGen wrapper versions
                    tet.regionlist = region_attributes_list
                    break
            return tet

        # Define fallback strategies with proper tolerance handling
        initial_switches = plc_switches
        if border_ids and 'p' not in initial_switches:
            initial_switches = 'p' + initial_switches

        # Strategy definitions
        strategies = [
            ('Initial', initial_switches, True),
            ('Robust + Tolerance', 'pT1e-12A', True),
            ('Very Permissive', 'pT1e-8A', True),
            ('No Constraints', 'pT1e-8A', False),  # Last resort: no edge constraints
        ]

        # Execute strategies
        for name, switches, use_edges in strategies:
            logger.info(f"Running TetGen with strategy: '{name}', switches: '{switches}'")
            try:
                tet = _create_tetgen_instance()
                
                # Optionally remove edge constraints for final fallback
                if not use_edges:
                    tet.edge_list = []
                    tet.edge_markers = []
                    logger.info("Removed edge constraints for final fallback")
                
                # Run TetGen
                tet.tetrahedralize(switches=switches)
                grid = tet.grid
                
                if grid is not None and grid.n_cells > 0:
                    logger.info(f"✓ TetGen succeeded with '{name}': {grid.n_cells} tetrahedra generated.")
                    return grid
                else:
                    logger.warning(f"TetGen ran with '{name}' but produced no tetrahedra.")
                    
            except Exception as e:
                logger.warning(f"TetGen failed with strategy '{name}': {e}")
        
        logger.error("All TetGen strategies failed. Surface data may have fundamental issues.")
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