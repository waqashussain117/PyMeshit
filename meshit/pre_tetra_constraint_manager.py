import numpy as np
from typing import List, Dict, Tuple

class PreTetraConstraintManager:
    """
    Builds per-surface constraint segments using the C++ MeshIt approach:
    1. Pre-insert triple points into intersection lines based on intersection IDs
    2. Segment intersection lines at points with type != "DEFAULT" 
    3. Create constraints from the segmented lines
    
    This matches the C++ workflow:
    - insert_int_triplepoints() -> AddPoint() to specific intersection lines
    - calculate_Constraints() -> segment at non-DEFAULT point types
    """

    def __init__(self) -> None:
        self.surface_constraints: Dict[int, List[dict]] = {}
        self.constraint_states:   Dict[Tuple[int,int], str] = {}
        self.surface_visibility:  Dict[int, bool]  = {}
        self._next_rgb_color      = [1.0, 0.0, 0.0]
        self._parent_gui = None

    def set_parent_gui(self, parent_gui):
        """Set reference to parent GUI for accessing global triple points"""
        self._parent_gui = parent_gui

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def generate_constraints_from_refine_data(self, datasets: list) -> None:
        """
        C++-style constraint generation with per-intersection-line IDs.

        For every surface we create
        • one HULL segment per hull edge  →  type="HULL"
        • one or more INTERSECTION segments per intersection line
          →  type="INTERSECTION",  line_id = <polyline index>

        The additional “line_id” key lets the GUI build several
        “Intersection Line n” groups instead of a single catch-all entry.
        """
        # reset containers
        self.surface_constraints.clear()
        self.constraint_states.clear()
        self.surface_visibility = {i: True for i in range(len(datasets))}

        # ---------------------------------------------------------------
        # 1)  prepare intersection-ID map and embed triple points
        # ---------------------------------------------------------------
        iid_map = self._build_intersection_id_mapping(datasets)
        self._insert_triple_points_into_intersections(datasets, iid_map)

        # ---------------------------------------------------------------
        # 2)  build constraint lists, surface by surface
        # ---------------------------------------------------------------
        for surf_idx, ds in enumerate(datasets):
            constraints = []

            # ---------- Hull segments -----------------------------------
            hull_raw = ds.get("hull_points")
            hull_pts = [] if hull_raw is None else [self._to_xyz(p) for p in hull_raw]
            if len(hull_pts) >= 3:
                for seg_id, seg in enumerate(self._create_hull_segments(hull_pts)):
                    constraints.append(
                        {"points": seg,
                         "type":   "HULL",
                         "segment_id": seg_id}
                    )

            # ---------- Intersection-line segments ----------------------
            processed_intersections = ds.get("processed_intersections", [])
            if not processed_intersections:
                processed_intersections = self._get_intersection_constraints(ds)

            for line_id, poly in enumerate(processed_intersections):
                segments = self._segment_at_special_points(poly)
                for seg_id, seg in enumerate(segments):
                    constraints.append(
                        {"points": seg,
                         "type":   "INTERSECTION",
                         "line_id": line_id,
                         "segment_id": seg_id}
                    )

            self.surface_constraints[surf_idx] = constraints

        # ---------------------------------------------------------------
        # 3)  default every new segment to the “SEGMENTS” state
        # ---------------------------------------------------------------
        for s_idx, lst in self.surface_constraints.items():
            for c_idx, _ in enumerate(lst):
                self.constraint_states[(s_idx, c_idx)] = "SEGMENTS"

        print(f"Generated constraints for {len(datasets)} surfaces")

    def _build_intersection_id_mapping(self, datasets: list) -> Dict[int, Tuple[int, int]]:
        """
        Build a mapping  {intersection_id  →  (dataset_index , local_polyline_index)}
        The code tries, in order,
        1.  stored_constraints[i]['intersection_id'  /  'id'  /  'line_id']
        2.  dataset['intersection_ids'][i]
        3.  (fallback) a synthetic ID = dataset_idx*1000 + local_idx
        """
        id_map: Dict[int, Tuple[int, int]] = {}

        for ds_idx, ds in enumerate(datasets):

            # ------- Format A :  stored_constraints ------
            if "stored_constraints" in ds:
                for local_idx, entry in enumerate(
                        [e for e in ds["stored_constraints"]
                         if e.get("type") == "intersection_line"]):

                    real_id = (entry.get("intersection_id")
                               or entry.get("id")
                               or entry.get("line_id"))
                    if real_id is None:
                        print(f"  WARNING: no intersection ID in Dataset {ds_idx}, "
                              f"polyline {local_idx}")
                        continue

                    id_map[real_id] = (ds_idx, local_idx)
                    print(f"  Intersection ID {real_id} → (ds={ds_idx}, local={local_idx})")

            # ------- Format B :  intersection_constraints (+ optional id list) ------
            elif "intersection_constraints" in ds:
                id_list = ds.get("intersection_ids", [])
                if id_list:
                    for local_idx, real_id in enumerate(id_list):
                        if local_idx < len(ds["intersection_constraints"]):
                            id_map[real_id] = (ds_idx, local_idx)
                            print(f"  Intersection ID {real_id} → (ds={ds_idx}, local={local_idx})")
                else:
                    # last-resort fallback – still keeps things unique
                    for local_idx, _ in enumerate(ds["intersection_constraints"]):
                        fake_id = ds_idx * 1000 + local_idx
                        id_map[fake_id] = (ds_idx, local_idx)
                        print(f"  Fallback ID {fake_id} → (ds={ds_idx}, local={local_idx})")

        print(f"Built intersection-ID map with {len(id_map)} entries\n")
        return id_map

    def _insert_triple_points_into_intersections(
            self,
            datasets: list,
            id_map: Dict[int, Tuple[int, int]]
    ) -> None:
        """
        Insert every triple point into **all** intersection polylines it belongs to.
        Strategy
        --------
        1.  If the triple point carries valid intersection IDs → insert by ID.
        2.  Otherwise (or if ID not found) → insert into the 3 closest
            polylines whose distance < 10 units.
        The actual insertion is done by _insert_triple_point_simple()
        which keeps vertex order.
        """
        print("Pre-inserting triple points …")

        # ---------- gather every TP (GUI + dataset) ----------
        all_tps = []          #  {point, intersection_ids, type}
        def _collect(tp_dict):
            pt_xyz = self._to_xyz(tp_dict.get("point"))
            if pt_xyz is None:
                return
            all_tps.append({
                "point": pt_xyz,
                "intersection_ids": (tp_dict.get("intersection_ids", [])
                                     or tp_dict.get("intersections", [])
                                     or tp_dict.get("intersection_list", [])),
                "type": "TRIPLE_POINT"
            })

        # from datasets
        for ds in datasets:
            for tp in ds.get("triple_points", []):
                if isinstance(tp, dict):
                    _collect(tp)

        # from GUI
        if hasattr(self, "_parent_gui") and hasattr(self._parent_gui, "triple_points"):
            for tp in self._parent_gui.triple_points:
                if isinstance(tp, dict):
                    _collect(tp)

        print(f"  collected {len(all_tps)} triple points\n")

        # ---------- insert them ----------
        inserted = 0

        for tp in all_tps:
            pt     = tp["point"]
            idlist = tp["intersection_ids"]
            done   = False

            # (1) ID-based
            for iid in idlist:
                target = id_map.get(iid)
                if target:
                    ds_idx, local_idx = target
                    if self._insert_triple_point_simple(datasets[ds_idx],
                                                        local_idx, pt, "TRIPLE_POINT"):
                        print(f"  ✓ TP {pt[:3]}  by-ID  →  ds {ds_idx}, poly {local_idx}")
                        inserted += 1
                        done = True

            # (2) proximity fallback
                       # (2) proximity fallback  – insert into *all* close lines
            if not done:
                for ds_idx, local_idx, dist in self._find_closest_intersections(datasets, pt):
                    if dist < 10.0:                       # distance threshold
                        if self._insert_triple_point_simple(
                                datasets[ds_idx], local_idx,
                                pt, "TRIPLE_POINT"):
                            print(f"  ✓ TP {pt[:3]}  by-dist {dist:.2f} "
                                  f"→ ds {ds_idx}, poly {local_idx}")
                            inserted += 1
                            # do NOT break: keep inserting into other copies  # stop after first success

        print(f"Inserted {inserted} triple-point instances\n")

    def _find_closest_intersections(
            self,
            datasets: list,
            point_xyz: list
    ) -> List[Tuple[int, int, float]]:
        """
        Return [(dataset_idx , local_idx , dist), …] sorted by distance
        from the triple-point to the *nearest* segment of each poly-line.
        """
        import numpy as np

        p = np.asarray(point_xyz, dtype=float)
        hits = []

        def _seg_dist(a, b):
            a_xyz = self._to_xyz(a)
            b_xyz = self._to_xyz(b)
            if a_xyz is None or b_xyz is None:
                return float('inf')
            a_np = np.asarray(a_xyz, dtype=float)
            b_np = np.asarray(b_xyz, dtype=float)
            ab = b_np - a_np
            L2 = np.dot(ab, ab)
            if L2 == 0.0:
                return np.linalg.norm(p - a_np)
            t = np.clip(np.dot(p - a_np, ab) / L2, 0.0, 1.0)
            proj = a_np + t * ab
            return np.linalg.norm(p - proj)

        for ds_idx, ds in enumerate(datasets):
            if "stored_constraints" in ds:
                polylines = [(i, entry["points"])
                             for i, entry in enumerate(ds["stored_constraints"])
                             if entry.get("type") == "intersection_line"]
            else:
                polylines = [(i, poly)
                             for i, poly in enumerate(ds.get("intersection_constraints", []))]

            for local_idx, poly in polylines:
                if len(poly) < 2:
                    continue
                d = min(_seg_dist(poly[i], poly[i + 1])
                        for i in range(len(poly) - 1))
                hits.append((ds_idx, local_idx, d))

        return sorted(hits, key=lambda t: t[2])

    def _point_to_line_segment_distance(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        """Calculate minimum distance from point to line segment"""
        segment_vec = seg_end - seg_start
        segment_length_sq = np.dot(segment_vec, segment_vec)
        
        if segment_length_sq < 1e-24:
            return np.linalg.norm(point - seg_start)
        
        # Project point onto line
        point_vec = point - seg_start
        t = np.clip(np.dot(point_vec, segment_vec) / segment_length_sq, 0.0, 1.0)
        
        closest_point = seg_start + t * segment_vec
        return np.linalg.norm(point - closest_point)

    def _insert_triple_point_simple(
            self,
            dataset: dict,
            local_intersection_idx: int,
            point_xyz: list,
            point_type: str
    ) -> bool:
        """
        Insert one triple-point into ONE poly-line while preserving geometry.

        ①  Find the segment whose orthogonal projection of TP is closest.
        ②  If that distance  <  adaptive_tol  → insert the projected point
            between those two vertices (never at index 0).
        ③  Otherwise, if TP is already closer than tol to an existing
            vertex, just mark that vertex as special (no duplicate).
        """

        import numpy as np

        line_pts = self._get_intersection_line_for_modification(dataset,
                                                                local_intersection_idx)
        if not line_pts or len(line_pts) < 2:
            return False

        tp = np.asarray(point_xyz, dtype=float)

        # ----- adaptive tolerance: 1 % of poly-line length, 1 mm minimum
        seg_len = np.sum([
            np.linalg.norm(np.asarray(p2['coord']) - np.asarray(p1['coord']))
            for p1, p2 in zip(line_pts[:-1], line_pts[1:])
        ])
        tol = max(1e-3, 0.01 * seg_len)

        # ----- helper: distance point – segment
        def _proj(p, a, b):
            a, b = map(np.asarray, (a, b))
            ab = b - a
            L2 = np.dot(ab, ab)
            if L2 == 0.0:
                return a.copy(), np.linalg.norm(p - a), 0.0
            t = np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0)
            proj = a + t * ab
            return proj, np.linalg.norm(p - proj), t

        # ----- choose best segment
        best_seg, best_d, best_proj = None, float("inf"), None
        for i in range(len(line_pts) - 1):
            proj, dist, _ = _proj(tp,
                                  line_pts[i]['coord'],
                                  line_pts[i + 1]['coord'])
            if dist < best_d:
                best_seg, best_d, best_proj = i, dist, proj

        # ----- already on an existing vertex?
        for pinfo in line_pts:
            if np.linalg.norm(tp - np.asarray(pinfo['coord'])) < tol:
                pinfo['type'] = point_type          # mark existing vertex
                return True

        # ----- insert between best_seg / best_seg+1  (only if inside tol)
        if best_seg is not None and best_d < tol:
            # avoid insertion as very first vertex
            insert_at = best_seg + 1
            if insert_at == 0:
                insert_at = 1
            line_pts.insert(insert_at,
                            {"coord": best_proj.tolist(), "type": point_type})
            return True

        return False

    def _get_intersection_line_for_modification(self, dataset: dict, local_intersection_idx: int) -> List[dict]:
        """
        Get intersection line points in a format that can be modified
        Returns reference to the actual points list for in-place modification
        """
        if "stored_constraints" in dataset:
            intersections = [entry for entry in dataset["stored_constraints"] 
                           if entry.get("type") == "intersection_line"]
            if local_intersection_idx < len(intersections):
                intersection = intersections[local_intersection_idx]
                # Create enhanced points if not exists
                if "enhanced_points" not in intersection:
                    intersection["enhanced_points"] = []
                    for pt in intersection.get("points", []):
                        intersection["enhanced_points"].append({
                            'coord': self._to_xyz(pt),
                            'type': 'DEFAULT'
                        })
                return intersection["enhanced_points"]
        
        elif "intersection_constraints" in dataset:
            # Create processed_intersections if not exists
            if "processed_intersections" not in dataset:
                dataset["processed_intersections"] = []
                for i, poly in enumerate(dataset["intersection_constraints"]):
                    enhanced_points = []
                    for pt in poly:
                        enhanced_points.append({
                            'coord': self._to_xyz(pt),
                            'type': 'DEFAULT'
                        })
                    dataset["processed_intersections"].append(enhanced_points)
            
            # Ensure we have enough entries
            while len(dataset["processed_intersections"]) <= local_intersection_idx:
                dataset["processed_intersections"].append([])
                
            return dataset["processed_intersections"][local_intersection_idx]
        
        return None

    def _segment_at_special_points(self, intersection_line: List[dict]) -> List[List[list]]:
        """
        Split a poly-line wherever a vertex's type != DEFAULT.

        Rule:  a special vertex is the *end* of the segment before the split
               and the *start* of the segment after it.
        """

        if len(intersection_line) < 2:
            return []

        # collect indices of split points (first / last always included)
        split_idx = {0, len(intersection_line) - 1}
        split_idx.update(i for i, v in enumerate(intersection_line)
                         if v['type'] != 'DEFAULT')

        ordered = sorted(split_idx)
        segments: List[List[list]] = []

        for a, b in zip(ordered[:-1], ordered[1:]):
            if b - a < 1:                # need at least 2 distinct points
                continue
            seg = [v['coord'] for v in intersection_line[a:b + 1]]
            if len(seg) >= 2 and seg[0] != seg[-1]:
                segments.append(seg)

        return segments
    def _create_hull_segments(self, hull: List[list]) -> List[List[list]]:
        """Create hull segments (C++ style - simple consecutive segments)"""
        if len(hull) < 3:
            return []
        
        segments = []
        for i in range(len(hull)):
            next_i = (i + 1) % len(hull)
            segments.append([hull[i], hull[next_i]])
        
        return segments

    def _get_intersection_constraints(self, dataset: dict) -> List[List[dict]]:
        """Fallback method to get intersection constraints in enhanced format"""
        constraints = []
        
        if "stored_constraints" in dataset:
            for entry in dataset["stored_constraints"]:
                if entry.get("type") == "intersection_line":
                    # Check if we have enhanced points (with triple points inserted)
                    if "enhanced_points" in entry:
                        constraints.append(entry["enhanced_points"])
                    else:
                        # Fallback to original points
                        enhanced_points = []
                        for pt in entry.get("points", []):
                            enhanced_points.append({
                                'coord': self._to_xyz(pt),
                                'type': 'DEFAULT'
                            })
                        constraints.append(enhanced_points)
        
        elif "intersection_constraints" in dataset:
            # Check if we have processed intersections (with triple points inserted)
            if "processed_intersections" in dataset:
                constraints = dataset["processed_intersections"]
            else:
                # Fallback to original format
                for poly in dataset["intersection_constraints"]:
                    enhanced_points = []
                    for pt in poly:
                        enhanced_points.append({
                            'coord': self._to_xyz(pt),
                            'type': 'DEFAULT'
                        })
                    constraints.append(enhanced_points)
        
        return constraints

    def generate_constraints_from_pre_tetra_data(self, datasets: list) -> None:
        """
        Alias for generate_constraints_from_refine_data for GUI compatibility
        This is called from the pre-tetra mesh tab
        """
        return self.generate_constraints_from_refine_data(datasets)

    def get_selected_surfaces(self) -> List[int]:
        return [i for i, v in self.surface_visibility.items() if v]

    def get_constraint_summary(self) -> dict:
        total = sum(len(c) for c in self.surface_constraints.values())
        active = sum(1 for s in self.constraint_states.values()
                     if s != "UNDEFINED")
        return {"total": total, "active": active}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_xyz(pt):
        """Extract [x,y,z] from various point formats, ignoring tags."""
        if pt is None:
            return None
            
        # Handle Vector3D objects (from intersection_utils)
        if hasattr(pt, 'x') and hasattr(pt, 'y') and hasattr(pt, 'z'):
            try:
                return [float(pt.x), float(pt.y), float(pt.z)]
            except (ValueError, TypeError):
                return None
                
        # Handle list/tuple formats
        if isinstance(pt, (list, tuple)):
            if len(pt) >= 3:
                try:
                    return [float(pt[0]), float(pt[1]), float(pt[2])]
                except (ValueError, TypeError):
                    return None
        elif hasattr(pt, '__len__') and len(pt) >= 3:
            try:
                return [float(pt[0]), float(pt[1]), float(pt[2])]
            except (ValueError, TypeError):
                return None
        return None