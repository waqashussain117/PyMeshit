# In tetra_mesh_utils.py

import numpy as np
import logging
import tetgen
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger("MeshIt-Workflow")


class TetrahedralMeshGenerator:
    """
    A utility class for generating tetrahedral meshes from surface data.
    
    This class follows the C++ MeshIt approach by directly using pre-computed
    conforming surface meshes to build the final PLC for TetGen.
    """
    
    def __init__(self, datasets: List[Dict], selected_surfaces: set, 
                 border_surface_indices: set, unit_surface_indices: set, 
                 fault_surface_indices: set, materials: List[Dict] = None,
                 surface_data: Dict = None):
        """
        Initialize the tetrahedral mesh generator.
        
        Args:
            datasets: List of surface datasets containing mesh data.
            selected_surfaces: Set of selected surface indices.
            border_surface_indices: Set of border surface indices.
            unit_surface_indices: Set of unit surface indices.
            fault_surface_indices: Set of fault surface indices.
            materials: List of material definitions with locations.
            surface_data: Dictionary of conforming mesh data {surface_idx: mesh_data}.
        """
        self.datasets = datasets
        self.selected_surfaces = selected_surfaces
        self.border_surface_indices = border_surface_indices
        self.unit_surface_indices = unit_surface_indices
        self.fault_surface_indices = fault_surface_indices
        self.materials = materials or []
        self.surface_data = surface_data or {}
        self.tetrahedral_mesh = None
        
        # PLC data containers
        self.plc_vertices = None
        self.plc_facets = None
        self.plc_facet_markers = None
        self.plc_regions = None
        self.plc_edge_constraints = None
        self.plc_edge_markers = None

    def generate_tetrahedral_mesh(self, tetgen_switches: str = "pq1.414aAY") -> Optional[pv.UnstructuredGrid]:
        """
        Generate tetrahedral mesh from the provided conforming surface data.
        """
        if not self.selected_surfaces:
            logger.warning("No surfaces selected for mesh generation.")
            return None
        
        logger.info("Generating tetrahedral mesh from pre-computed conforming PLC...")
        
        try:
            # Step 1: Directly use the conforming mesh data to build the final PLC.
            self._build_plc_from_precomputed_meshes()
            
            if self.plc_vertices is None or len(self.plc_vertices) == 0:
                logger.error("PLC assembly failed: No vertices found.")
                return None
            
            # Step 2: Create a TetGen object.
            tet = tetgen.TetGen(self.plc_vertices, self.plc_facets, self.plc_facet_markers)
            
            # Step 3: Add edge constraints.
            if self.plc_edge_constraints is not None and len(self.plc_edge_constraints) > 0:
                tet.edge_list = self.plc_edge_constraints.tolist()
                tet.edge_marker_list = self.plc_edge_markers.tolist()
                logger.info(f"Added {len(self.plc_edge_constraints)} intersection edge constraints to TetGen.")
            
            # Step 4: Add material seed points.
            if self.materials:
                self.plc_regions = self._prepare_material_regions()
                if self.plc_regions:
                    tet.regions = self.plc_regions
                    logger.info(f"Added {len(self.plc_regions)} material seed points to TetGen.")
            
            # Step 5: Run TetGen.
            logger.info(f"Running TetGen with switches: '{tetgen_switches}'")
            tet.tetrahedralize(switches=tetgen_switches)
            grid = tet.grid
            
            if grid is None or grid.n_cells == 0:
                logger.error("TetGen ran but produced no tetrahedra.")
                self._export_plc_for_debugging()
                return self._run_tetgen_fallback_strategies(tetgen_switches)
            
            logger.info(f"✓ TetGen succeeded: {grid.n_cells} tetrahedra generated.")
            self.tetrahedral_mesh = grid
            return grid
            
        except Exception as e:
            logger.error(f"TetGen execution failed: {e}", exc_info=True)
            self._export_plc_for_debugging()
            return self._run_tetgen_fallback_strategies(tetgen_switches)
    
    def _build_plc_from_precomputed_meshes(self):
        """
        Builds the final PLC by combining pre-computed conforming meshes.
        FIXED: Fault surfaces treated as internal constraints, not boundary facets.
        """
        logger.info("Building final PLC from pre-computed conforming meshes...")

        key_to_global_idx = {}
        global_vertices = []
        global_facets = []
        global_facet_markers = []
        edge_constraints = set()

        # CRITICAL FIX: Only boundary surfaces define the domain boundary
        boundary_surfaces = (self.border_surface_indices | self.unit_surface_indices) & self.selected_surfaces
        fault_surfaces = self.fault_surface_indices & self.selected_surfaces

        logger.info(f"Processing {len(boundary_surfaces)} boundary surfaces, {len(fault_surfaces)} fault surfaces as internal constraints")

        # 1. Process boundary surfaces (borders + units) - these define the domain
        for s_idx in boundary_surfaces:
            if s_idx not in self.surface_data:
                logger.warning(f"Boundary surface {s_idx} selected but has no conforming mesh data. Skipping.")
                continue
            
            conforming_mesh = self.surface_data[s_idx]
            local_vertices = conforming_mesh.get('vertices')
            local_triangles = conforming_mesh.get('triangles')

            if local_vertices is None or local_triangles is None or len(local_vertices) == 0:
                continue

            local_to_global_map = {}
            for local_idx, vertex in enumerate(local_vertices):
                key = (round(vertex[0], 9), round(vertex[1], 9), round(vertex[2], 9))
                global_idx = key_to_global_idx.get(key)
                if global_idx is None:
                    global_idx = len(global_vertices)
                    key_to_global_idx[key] = global_idx
                    global_vertices.append(list(vertex))
                local_to_global_map[local_idx] = global_idx

            for tri in local_triangles:
                global_tri = [local_to_global_map.get(v_idx) for v_idx in tri[:3]]
                if all(v is not None for v in global_tri) and len(set(global_tri)) == 3:
                    v0, v1, v2 = [global_vertices[i] for i in global_tri]
                    area = 0.5 * np.linalg.norm(np.cross(np.array(v1) - np.array(v0), np.array(v2) - np.array(v0)))
                    if area > 1e-12:
                        global_facets.append(global_tri)
                        global_facet_markers.append(s_idx)

        # 2. Process fault surfaces - add vertices but not facets (they become internal constraints)
        fault_edge_constraints = set()
        for s_idx in fault_surfaces:
            if s_idx not in self.surface_data:
                logger.warning(f"Fault surface {s_idx} selected but has no conforming mesh data. Skipping.")
                continue
            
            conforming_mesh = self.surface_data[s_idx]
            local_vertices = conforming_mesh.get('vertices')
            local_triangles = conforming_mesh.get('triangles')

            if local_vertices is None or local_triangles is None or len(local_vertices) == 0:
                continue

            # Add fault vertices to global pool
            local_to_global_map = {}
            for local_idx, vertex in enumerate(local_vertices):
                key = (round(vertex[0], 9), round(vertex[1], 9), round(vertex[2], 9))
                global_idx = key_to_global_idx.get(key)
                if global_idx is None:
                    global_idx = len(global_vertices)
                    key_to_global_idx[key] = global_idx
                    global_vertices.append(list(vertex))
                local_to_global_map[local_idx] = global_idx

            # Convert fault triangles to edge constraints (internal discontinuities)
            for tri in local_triangles:
                global_tri = [local_to_global_map.get(v_idx) for v_idx in tri[:3]]
                if all(v is not None for v in global_tri) and len(set(global_tri)) == 3:
                    # Add triangle edges as constraints instead of the triangle as a facet
                    for i in range(3):
                        v1_idx = global_tri[i]
                        v2_idx = global_tri[(i + 1) % 3]
                        if v1_idx != v2_idx:
                            edge_tuple = tuple(sorted((v1_idx, v2_idx)))
                            fault_edge_constraints.add(edge_tuple)

        logger.info(f"Added {len(fault_edge_constraints)} fault edge constraints from {len(fault_surfaces)} fault surfaces")

        # 3. Re-map intersection line constraints with enhanced precision
        constraint_point_failures = 0
        for s_idx in self.selected_surfaces:
            if s_idx >= len(self.datasets): continue
            
            dataset = self.datasets[s_idx]
            for constraint in dataset.get("stored_constraints", []):
                if constraint.get("type") == "intersection_line":
                    points = constraint.get("points", [])
                    if len(points) < 2: continue
                    
                    gidx_line = []
                    for p in points:
                        # Enhanced precision matching for constraint points
                        key = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
                        gidx = key_to_global_idx.get(key)
                        if gidx is not None:
                            gidx_line.append(gidx)
                        else:
                            # Try with slightly relaxed precision for constraint points
                            found = False
                            for existing_key, existing_gidx in key_to_global_idx.items():
                                if (abs(existing_key[0] - key[0]) < 1e-8 and 
                                    abs(existing_key[1] - key[1]) < 1e-8 and 
                                    abs(existing_key[2] - key[2]) < 1e-8):
                                    gidx_line.append(existing_gidx)
                                    found = True
                                    break
                            
                            if not found:
                                constraint_point_failures += 1
                                if constraint_point_failures <= 5:  # Only log first few failures
                                    logger.warning(f"Constraint point {key} not found in global vertex pool (surface {s_idx}).")
                    
                    if len(gidx_line) >= 2:
                        for i in range(len(gidx_line) - 1):
                            p1_gidx, p2_gidx = gidx_line[i], gidx_line[i+1]
                            if p1_gidx != p2_gidx:
                                edge_tuple = tuple(sorted((p1_gidx, p2_gidx)))
                                edge_constraints.add(edge_tuple)

        # Combine intersection and fault edge constraints
        all_edge_constraints = edge_constraints | fault_edge_constraints

        if constraint_point_failures > 5:
            logger.warning(f"Total constraint point failures: {constraint_point_failures} (only first 5 logged)")

        # Finalize PLC data
        self.plc_vertices = np.asarray(global_vertices, dtype=np.float64)
        self.plc_facets = np.asarray(global_facets, dtype=np.int32)
        self.plc_facet_markers = np.asarray(global_facet_markers, dtype=np.int32)
        self.plc_edge_constraints = np.asarray(list(all_edge_constraints), dtype=np.int32) if all_edge_constraints else np.array([], dtype=np.int32).reshape(0, 2)
        self.plc_edge_markers = np.arange(1, len(self.plc_edge_constraints) + 1, dtype=np.int32) if len(all_edge_constraints) > 0 else np.array([], dtype=np.int32)

        logger.info(f"Final PLC created: {len(self.plc_vertices)} vertices, {len(self.plc_facets)} facets (boundary only), {len(self.plc_edge_constraints)} edge constraints ({len(edge_constraints)} intersection + {len(fault_edge_constraints)} fault).")

    def _prepare_material_regions(self) -> List[List[float]]:
        region_attributes_list = []
        if not self.materials:
            if self.plc_vertices is not None and len(self.plc_vertices) > 0:
                bounds_min = np.min(self.plc_vertices, axis=0)
                bounds_max = np.max(self.plc_vertices, axis=0)
                center = (bounds_min + bounds_max) / 2.0
                region_attributes_list.append([center[0], center[1], center[2], 1, -1])
                logger.info("No materials defined. Using default material region at PLC center.")
        else:
            for material_idx, material in enumerate(self.materials):
                locations = material.get('locations', [])
                material_attribute = material.get('attribute', material_idx + 1)
                for loc in locations:
                    if len(loc) >= 3:
                        region_attributes_list.append([float(loc[0]), float(loc[1]), float(loc[2]), int(material_attribute), -1])
            logger.info(f"Prepared {len(region_attributes_list)} material regions from {len(self.materials)} materials.")
        return region_attributes_list

    def _run_tetgen_fallback_strategies(self, original_switches: str) -> Optional[pv.UnstructuredGrid]:
        logger.warning("Initial TetGen failed. Trying fallback strategies...")
        fallback_switches = [
            "pq1.2aAY",  # C++ command line style
            "pAY",       # C++ GUI style
            "pA",        # Basic (what worked before)
            "pzQ"        # Last resort
        ]
        for switches in fallback_switches:
            try:
                logger.warning(f"Trying fallback TetGen switches: '{switches}'")
                tet = tetgen.TetGen(self.plc_vertices, self.plc_facets, self.plc_facet_markers)
                if self.plc_edge_constraints is not None and len(self.plc_edge_constraints) > 0:
                    tet.edge_list = self.plc_edge_constraints.tolist()
                    tet.edge_marker_list = self.plc_edge_markers.tolist()
                if self.plc_regions:
                    tet.regions = self.plc_regions
                tet.tetrahedralize(switches=switches)
                grid = tet.grid
                if grid is not None and grid.n_cells > 0:
                    logger.info(f"✓ Fallback TetGen succeeded with '{switches}': {grid.n_cells} tetrahedra")
                    self.tetrahedral_mesh = grid
                    return grid
                else:
                    logger.warning(f"Fallback switches '{switches}' produced no tetrahedra.")
            except Exception as e:
                logger.warning(f"Fallback switches '{switches}' also failed: {e}")
        logger.error("All TetGen strategies failed. The input PLC likely has severe issues.")
        return None
    
    def _export_plc_for_debugging(self):
        try:
            logger.info("Exporting PLC to debug_plc.vtm for inspection...")
            mesh = pv.PolyData(self.plc_vertices, faces=np.hstack((np.full((len(self.plc_facets), 1), 3), self.plc_facets)))
            mesh.cell_data['surface_id'] = self.plc_facet_markers
            
            multi_block = pv.MultiBlock()
            multi_block.append(mesh, "Facets")

            if self.plc_edge_constraints is not None and len(self.plc_edge_constraints) > 0:
                lines = []
                for edge in self.plc_edge_constraints:
                    lines.extend([2, edge[0], edge[1]])
                edge_mesh = pv.PolyData(self.plc_vertices, lines=np.array(lines))
                multi_block.append(edge_mesh, "Constraints")

            multi_block.save("debug_plc.vtm")
            logger.info("PLC debug file saved as debug_plc.vtm")
        except Exception as e:
            logger.error(f"Failed to export PLC debug files: {e}")

    def export_mesh(self, file_path: str, mesh_data: Optional[Dict] = None) -> bool:
        if mesh_data is None: mesh_data = self.tetrahedral_mesh
        if not mesh_data:
            logger.error("No tetrahedral mesh to export")
            return False
        try:
            if isinstance(mesh_data, pv.UnstructuredGrid):
                mesh_data.save(file_path)
                logger.info(f"Tetrahedral mesh exported to: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
        return False

    def get_mesh_statistics(self, mesh_data: Optional[Dict] = None) -> Dict[str, Union[int, float]]:
        if mesh_data is None: mesh_data = self.tetrahedral_mesh
        if not mesh_data: return {}
        stats = {}
        if isinstance(mesh_data, pv.UnstructuredGrid):
            stats['n_vertices'] = mesh_data.n_points
            stats['n_tetrahedra'] = mesh_data.n_cells
            stats['volume'] = float(mesh_data.volume) if hasattr(mesh_data, 'volume') else 0.0
        return stats


# *** FIX: Move this function OUTSIDE the class definition ***
def create_tetrahedral_mesh(datasets: List[Dict], selected_surfaces: set, 
                           border_surface_indices: set, unit_surface_indices: set, 
                           fault_surface_indices: set, materials: List[Dict] = None,
                           tetgen_switches: str = "pq1.414aAY") -> Optional[Dict]:
    """
    Convenience function to create a tetrahedral mesh.
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