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
            
            # Step 4: Add material seed points using C++ MeshIt approach
            if self.materials:
                # Separate faults (2D surface materials) from units (3D volumetric materials)
                volumetric_regions, surface_materials = self._prepare_materials_cpp_style()
                
                if volumetric_regions:
                    # Only add VOLUMETRIC materials (units/formations) as 3D regions
                    for i, region in enumerate(volumetric_regions):
                        region_id = int(region[3])  # material attribute
                        point = (float(region[0]), float(region[1]), float(region[2]))
                        max_vol = float(region[4]) if region[4] > 0 else 0.0  # TetGen expects 0.0 for no constraint
                        
                        tet.add_region(region_id, point, max_vol)
                        logger.debug(f"Added 3D region {region_id}: point={point}, max_vol={max_vol}")
                    
                    logger.info(f"✓ C++ Style: Added {len(volumetric_regions)} 3D regions (units/formations) to TetGen")
                    
                if surface_materials:
                    logger.info(f"✓ C++ Style: {len(surface_materials)} 2D materials (faults) handled as surface constraints")
                    # Note: Faults are already included in the surface triangulation as constraints
                    # They don't need separate 3D region seeds - this matches C++ behavior
                    
                # Store for attribute assignment later
                self.plc_regions = volumetric_regions
                self.surface_materials = surface_materials
            
            # Step 5: Run TetGen.
            logger.info(f"Running TetGen with switches: '{tetgen_switches}'")
            # Ensure region attributes are enabled when we have volumetric materials (C++ style)
            if self.materials and hasattr(self, 'plc_regions') and self.plc_regions:
                # Enable region attributes using the regionattrib parameter (equivalent to '-A' switch)
                if 'A' in tetgen_switches:
                    # Capture the returned attributes when regions are defined
                    nodes, elements, attributes = tet.tetrahedralize(switches=tetgen_switches, regionattrib=1.0)
                else:
                    # Add 'A' switch if not present
                    modified_switches = tetgen_switches + 'A'
                    logger.info(f"Added 'A' switch for region attributes: '{modified_switches}'")
                    nodes, elements, attributes = tet.tetrahedralize(switches=modified_switches, regionattrib=1.0)
                
                # Apply the material attributes to the grid
                grid = tet.grid
                if attributes is not None and len(attributes) > 0:
                    grid.cell_data['MaterialID'] = attributes.astype(int)
                    import numpy as np
                    unique_materials = np.unique(attributes.astype(int))
                    logger.info(f"✓ Applied TetGen material attributes directly: {unique_materials}")
                else:
                    logger.warning("TetGen returned no material attributes despite having regions")
            else:
                tet.tetrahedralize(switches=tetgen_switches)
                grid = tet.grid
            
            # Step 6: Check if material attributes were successfully applied
            # If not, fall back to manual assignment
            if grid is not None and grid.n_cells > 0:
                if 'MaterialID' not in grid.cell_data or len(grid.cell_data['MaterialID']) == 0:
                    logger.info("No MaterialID found in mesh - will use manual assignment")
                else:
                    # Material attributes were successfully applied from TetGen
                    logger.info("✓ Material attributes successfully obtained from TetGen")
            
            if grid is None or grid.n_cells == 0:
                logger.error("TetGen ran but produced no tetrahedra.")
                self._export_plc_for_debugging()
                return self._run_tetgen_fallback_strategies(tetgen_switches)
            
            logger.info(f"✓ TetGen succeeded: {grid.n_cells} tetrahedra generated.")
            self.tetrahedral_mesh = grid
            
            # ✅ CRITICAL: Store TetGen object to access constraint surface triangles (C++ style)
            self.tetgen_object = tet
            logger.debug(f"Stored TetGen object for constraint surface access")
            
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

        # 2. ✅ CRITICAL FIX: Process fault surfaces as constraint facets (C++ style) 
        # This preserves fault triangles for visualization instead of converting to edges
        fault_edge_constraints = set()
        fault_surface_marker_mapping = {}  # Track which marker corresponds to which fault surface
        
        for fault_idx, s_idx in enumerate(fault_surfaces):
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

            # ✅ C++ STYLE: Add fault triangles as constraint facets with unique markers
            fault_marker = 1000 + fault_idx  # Use high marker values to distinguish from boundaries
            fault_surface_marker_mapping[fault_marker] = s_idx
            
            for tri in local_triangles:
                global_tri = [local_to_global_map.get(v_idx) for v_idx in tri[:3]]
                if all(v is not None for v in global_tri) and len(set(global_tri)) == 3:
                    # ✅ Add triangle as constraint facet (preserves triangulation for C++ style visualization)
                    global_facets.append(global_tri)
                    global_facet_markers.append(fault_marker)
                    
                    # Also add edges as constraints for TetGen edge list
                    for i in range(3):
                        v1_idx = global_tri[i]
                        v2_idx = global_tri[(i + 1) % 3]
                        if v1_idx != v2_idx:
                            edge_tuple = tuple(sorted((v1_idx, v2_idx)))
                            fault_edge_constraints.add(edge_tuple)

        self.fault_surface_markers = fault_surface_marker_mapping  # Store for visualization
        logger.info(f"✅ Added {len([m for m in global_facet_markers if m >= 1000])} fault constraint facets from {len(fault_surfaces)} fault surfaces")
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
                # FIX: Use 0 for volume constraint (no constraint), not -1
                region_attributes_list.append([center[0], center[1], center[2], 1, 0])
                logger.info("No materials defined. Using default material region at PLC center.")
        else:
            for material_idx, material in enumerate(self.materials):
                locations = material.get('locations', [])
                material_attribute = material.get('attribute', material_idx + 1)
                for loc in locations:
                    if len(loc) >= 3:
                        # FIX: Use 0 for volume constraint (no constraint), not -1
                        # This matches C++ geometry.cpp: in.regionlist[currentRegion*5+4]=0;
                        region_attributes_list.append([float(loc[0]), float(loc[1]), float(loc[2]), int(material_attribute), 0])
            logger.info(f"Prepared {len(region_attributes_list)} material regions from {len(self.materials)} materials.")
        return region_attributes_list

    def _prepare_materials_cpp_style(self) -> tuple:
        """
        Prepare materials following EXACT C++ MeshIt approach:
        - ONLY FORMATIONS become 3D volumetric regions (units/formations)
        - FAULTS are ONLY surface constraints (NO volumetric regions)
        - Material IDs MUST be sequential indices (0, 1, 2...) like C++ Mats array
        - This matches C++ where faults are facetmarkerlist[], formations are regionlist[]
        
        Returns:
            tuple: (volumetric_regions, surface_materials)
        """
        volumetric_regions = []
        surface_materials = []
        
        if not self.materials:
            # Default fallback: create one volumetric region at PLC center with material ID = 0
            if self.plc_vertices is not None and len(self.plc_vertices) > 0:
                bounds_min = np.min(self.plc_vertices, axis=0)
                bounds_max = np.max(self.plc_vertices, axis=0)
                center = (bounds_min + bounds_max) / 2.0
                volumetric_regions.append([center[0], center[1], center[2], 0, 0])  # Material ID = 0
                logger.info("No materials defined. Using default volumetric material region at PLC center with ID=0")
        else:
            # C++ Style: ALL materials get volumetric regions with SEQUENTIAL indices (0,1,2...)
            # CRITICAL: Sort materials by their attribute to ensure sequential ordering!
            sorted_materials = sorted(self.materials, key=lambda m: m.get('attribute', 0))
            
            for material in sorted_materials:
                material_name = material.get('name', '').lower()
                material_type = material.get('type', 'FORMATION')
                locations = material.get('locations', [])
                material_attribute = material.get('attribute', 0)  # Use material's attribute as ID
                
                # Check if this is a fault
                is_fault = (material_type == 'FAULT' or 
                           any(keyword in material_name for keyword in ['fault', 'fracture', 'crack', 'fissure']))
                
                if is_fault:
                    # CRITICAL: Faults are ONLY surface constraints - NO volumetric regions!
                    surface_materials.append(material)
                    logger.debug(f"Material {material_attribute} '{material_name}' -> FAULT (surface constraint only)")
                    continue
                
                # ONLY formations/units get volumetric regions
                for loc in locations:
                    if len(loc) >= 3:
                        # CRITICAL: Use the material's attribute directly (already sequential 0,1,2...)
                        volumetric_regions.append([float(loc[0]), float(loc[1]), float(loc[2]), material_attribute, 0])
                        logger.debug(f"Added 3D region: '{material.get('name')}' with C++ style ID={material_attribute}")
        
        max_material_id = max([int(region[3]) for region in volumetric_regions]) if volumetric_regions else -1
        logger.info(f"✓ TRUE C++ Style: {len(volumetric_regions)} volumetric regions (formations only, TetGen indices 0-{max_material_id})")
        logger.info(f"✓ TRUE C++ Style: {len(surface_materials)} surface materials (faults only, surface constraints)")
        
        return volumetric_regions, surface_materials

    def _run_tetgen_fallback_strategies(self, original_switches: str) -> Optional[pv.UnstructuredGrid]:
        logger.warning("Initial TetGen failed. Trying fallback strategies...")
        fallback_switches = [
            "pq1.2aAY",  # C++ command line style with materials
            "pq1.2aA",   # Remove Y but keep A for materials  
            "pAY",       # C++ GUI style
            "pA",        # Basic with materials
            "pzQ"        # Last resort (no materials)
        ]
        for switches in fallback_switches:
            try:
                logger.warning(f"Trying fallback TetGen switches: '{switches}'")
                tet = tetgen.TetGen(self.plc_vertices, self.plc_facets, self.plc_facet_markers)
                if self.plc_edge_constraints is not None and len(self.plc_edge_constraints) > 0:
                    tet.edge_list = self.plc_edge_constraints.tolist()
                    tet.edge_marker_list = self.plc_edge_markers.tolist()
                # Use add_region() method for fallback strategies too (C++ style - only volumetric regions)
                if hasattr(self, 'plc_regions') and self.plc_regions:
                    for region in self.plc_regions:
                        region_id = int(region[3])  # material attribute
                        point = (float(region[0]), float(region[1]), float(region[2]))
                        max_vol = float(region[4]) if region[4] > 0 else 0.0
                        tet.add_region(region_id, point, max_vol)
                # Enable region attributes for fallback if 'A' switch is present and we have regions
                if hasattr(self, 'plc_regions') and self.plc_regions and 'A' in switches:
                    nodes, elements, attributes = tet.tetrahedralize(switches=switches, regionattrib=1.0)
                    grid = tet.grid
                    # Apply material attributes from fallback too
                    if attributes is not None and len(attributes) > 0:
                        grid.cell_data['MaterialID'] = attributes.astype(int)
                        import numpy as np
                        unique_materials = np.unique(attributes.astype(int))
                        logger.info(f"✓ Fallback: Applied TetGen material attributes: {unique_materials}")
                else:
                    tet.tetrahedralize(switches=switches)
                    grid = tet.grid
                if grid is not None and grid.n_cells > 0:
                    logger.info(f"✓ Fallback TetGen succeeded with '{switches}': {grid.n_cells} tetrahedra")
                    
                    # Check if material attributes were applied in fallback
                    if 'MaterialID' in grid.cell_data and len(grid.cell_data['MaterialID']) > 0:
                        logger.info("✓ Fallback: Material attributes successfully obtained from TetGen")
                    else:
                        logger.info("Fallback: No material attributes - will need manual assignment")
                    
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