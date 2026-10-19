"""
qc_audit_tool.py
----------------
Quality Assurance Audit & Correction Tool for Building-Road Overlaps.
- Supports single/multiple .zip archives and .shp inputs.
- Executes high-performance spatial correction (trimming, patch moving, sliver purge).
- Exports comprehensive audit Excel reports with full polygon attributes:
    1. Summary Dashboard (Audit Ticket Metrics)
    2. Total Overlaps (All detected conflicts before correction)
    3. Fixed Using Automation (Trimmed & Shifted polygons)
    4. Fixes Pending - Manual Review (Unresolved conflicts requiring human intervention)
    5. Deleted Slivers (Artifacts < 3m²)
"""

import os
import sys
import zipfile
import tempfile
import shutil
import warnings
import concurrent.futures
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import shapely.affinity
from shapely.geometry import shape, MultiPolygon, Polygon, box
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")

try:
    import pyogrio
    HAS_PYOGRIO = True
except ImportError:
    HAS_PYOGRIO = False

# ── Processing Constants ───────────────────────────────────────────────────────
ROAD_BUFFER_M      = 1.0
SMALL_AREA_THRESH  = 3.0
HIGH_OVERLAP_RATIO = 0.20


# ── Geometry & Clean-up Helpers ───────────────────────────────────────────────

def clean_coords(coords):
    """Recursively checks if coordinate rings are closed."""
    if not isinstance(coords, (list, tuple)):
        return coords
    if len(coords) > 0 and isinstance(coords[0], (tuple, list)) and isinstance(coords[0][0], (int, float)):
        if coords[0] != coords[-1]:
            return list(coords) + [coords[0]]
        return coords
    return [clean_coords(c) for c in coords]


def extract_polygons(geom):
    """Extracts valid Polygon/MultiPolygon from mixed geometry collections."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if geom.geom_type == 'GeometryCollection' or hasattr(geom, 'geoms'):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        return unary_union(polys) if polys else None
    return None


def largest_polygon(geom):
    """Returns the single largest Polygon from a MultiPolygon to discard slivers."""
    geom = extract_polygons(geom)
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return None


def get_compactness(geom):
    """Calculates Polsby-Popper compactness score."""
    if geom is None or geom.is_empty or geom.length == 0:
        return 0.0
    return (4 * np.pi * geom.area) / (geom.length ** 2)


def is_shape_preserved(orig_geom, new_geom):
    """Checks validity, minimal area thresholds, area loss bounds, and compactness."""
    if not new_geom or new_geom.is_empty or not new_geom.is_valid:
        return False
    if new_geom.area < SMALL_AREA_THRESH:
        return False
    if new_geom.area < 0.75 * orig_geom.area:
        return False

    orig_c = get_compactness(orig_geom)
    new_c = get_compactness(new_geom)
    if orig_c > 0:
        if new_c < 0.5 * orig_c or new_c > 2.0 * orig_c:
            return False
    return True


def build_patches_fast(movable_polys):
    """Groups touching polygons into connected components using STRtree + BFS."""
    if movable_polys.empty:
        return []

    geoms = movable_polys.geometry.values
    idx_map = movable_polys.index.values
    tree = shapely.STRtree(geoms)

    left, right = tree.query(geoms, predicate="intersects")
    n = len(geoms)
    adj = [[] for _ in range(n)]
    for l, r in zip(left, right):
        if l != r:
            adj[l].append(r)

    visited = np.zeros(n, dtype=bool)
    patches = []

    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            patches.append([idx_map[x] for x in queue])

    return patches


def is_move_valid_fast(patch_union, dx, dy, local_roads_union, local_other_union):
    """Validates if moving a patch clears overlaps without colliding with other features."""
    shifted = shapely.affinity.translate(patch_union, xoff=dx, yoff=dy)
    if not shifted.is_valid:
        return False

    if local_roads_union is not None and not local_roads_union.is_empty:
        if shifted.intersects(local_roads_union):
            if shifted.intersection(local_roads_union).area > 0.01:
                return False

    if local_other_union is not None and not local_other_union.is_empty:
        if shifted.intersects(local_other_union):
            if shifted.intersection(local_other_union).area > 0.01:
                return False

    return True


# ── File & ZIP Ingestion Helpers ──────────────────────────────────────────────

def resolve_shapefile(path, extract_dir):
    """
    Extracts a zip archive if needed and returns the path to the inner .shp file.
    If already a .shp file, returns the path as is.
    """
    if path.lower().endswith(".zip"):
        subfolder = os.path.join(extract_dir, os.path.splitext(os.path.basename(path))[0])
        os.makedirs(subfolder, exist_ok=True)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(subfolder)

        # Locate .shp inside
        for root, _, files in os.walk(subfolder):
            for file in files:
                if file.lower().endswith(".shp") and not file.startswith("._"):
                    return os.path.join(root, file)
        raise FileNotFoundError(f"No .shp file found inside {path}")
    elif path.lower().endswith(".shp"):
        return path
    else:
        raise ValueError(f"Unsupported file format: {path}. Expected .shp or .zip.")


def robust_load_shp(path, log_callback):
    """Reads shapefile natively via pyogrio/fiona with vectorized GEOS repairs."""
    filename = os.path.basename(path)
    try:
        if HAS_PYOGRIO:
            gdf = pyogrio.read_dataframe(path)
        else:
            gdf = gpd.read_file(path, engine="fiona")

        invalid_mask = ~gdf.geometry.is_valid
        num_invalid = invalid_mask.sum()
        if num_invalid > 0:
            log_callback(f"  -> Found {num_invalid} invalid geometries in {filename}. Repairing...")
            gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
        return gdf
    except Exception as e:
        log_callback(f"  -> Direct read failed: {e}. Attempting manual ring closure...")
        import fiona
        data = []
        with fiona.open(path, 'r') as source:
            crs = source.crs
            for feature in source:
                geom_dict = feature['geometry']
                properties = dict(feature['properties'])
                if geom_dict is None:
                    continue
                try:
                    geom = shape(geom_dict)
                except Exception:
                    try:
                        geom_dict['coordinates'] = clean_coords(geom_dict['coordinates'])
                        geom = shape(geom_dict)
                    except Exception:
                        continue
                if not geom.is_valid:
                    geom = shapely.make_valid(geom)
                properties['geometry'] = geom
                data.append(properties)

        if not data:
            raise ValueError(f"Could not recover geometries from {filename}")
        return gpd.GeoDataFrame(data, crs=crs)


def load_multiple_polygons(paths, extract_dir, log_callback):
    """Loads and merges multiple polygon shapefiles or zip files into a single GeoDataFrame."""
    gdfs = []
    base_crs = None

    for p in paths:
        log_callback(f"Processing polygon source: {os.path.basename(p)}...")
        resolved_path = resolve_shapefile(p, extract_dir)
        gdf = robust_load_shp(resolved_path, log_callback)

        if gdf.empty:
            continue

        if base_crs is None:
            base_crs = gdf.crs
        elif gdf.crs != base_crs and base_crs is not None:
            gdf = gdf.to_crs(base_crs)

        gdf["_SOURCE_FILE"] = os.path.basename(p)
        gdfs.append(gdf)

    if not gdfs:
        raise ValueError("No valid polygon records found across provided input files.")

    merged_gdf = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(merged_gdf, crs=base_crs)


# ── Patch Worker Task ─────────────────────────────────────────────────────────

def process_patch_task(
    patch_indices,
    poly_to_roads,
    polys_geom_dict,
    buffered_roads_dict,
    retained_polys_geoms,
    retained_tree,
    road_tree_buffered,
    buffered_roads_geoms,
    retained_idx_map,
    enable_trim,
    enable_move
):
    """Processes a connected component of polygons."""
    results = {}
    overlapping_polys = [i for i in patch_indices if i in poly_to_roads]

    if not overlapping_polys:
        for i in patch_indices:
            results[i] = (polys_geom_dict[i], "unchanged", 0.0, "No conflict")
        return results, 0

    poly_overlap_info = {}
    for i in overlapping_polys:
        geom = polys_geom_dict[i]
        r_geoms = [buffered_roads_dict[r] for r in poly_to_roads[i] if r in buffered_roads_dict]
        if not r_geoms:
            continue
        local_r_union = shapely.ops.unary_union(r_geoms)
        if not local_r_union.is_valid:
            local_r_union = shapely.make_valid(local_r_union)

        inter = geom.intersection(local_r_union)
        if inter.area > 0.001:
            ratio = inter.area / geom.area
            poly_overlap_info[i] = {
                'overlap_area': inter.area,
                'ratio': ratio,
                'road_union': local_r_union
            }

    if not poly_overlap_info:
        for i in patch_indices:
            results[i] = (polys_geom_dict[i], "unchanged", 0.0, "No conflict")
        return results, 0

    num_overlaps = len(poly_overlap_info)
    is_patch = len(patch_indices) > 1
    patch_needs_move = is_patch and num_overlaps > 3 and enable_move
    move_successful = False

    patch_geom_dict = {i: polys_geom_dict[i] for i in patch_indices}
    patch_union = shapely.ops.unary_union(list(patch_geom_dict.values()))
    if not patch_union.is_valid:
        patch_union = shapely.make_valid(patch_union)

    # 1. Try moving patch
    if patch_needs_move:
        search_bounds = patch_union.buffer(10).bounds
        query_box = box(*search_bounds)

        local_road_idx = road_tree_buffered.query(query_box, predicate="intersects")
        r_union = shapely.ops.unary_union(buffered_roads_geoms[local_road_idx]) if len(local_road_idx) > 0 else None

        local_other_idx = retained_tree.query(query_box, predicate="intersects")
        local_other_geoms = [
            retained_polys_geoms[idx]
            for idx in local_other_idx
            if retained_idx_map[idx] not in patch_indices
        ]
        o_union = shapely.ops.unary_union(local_other_geoms) if local_other_geoms else None

        dx, dy = None, None
        angles = np.deg2rad(np.arange(0, 360, 22.5))
        for dist in np.arange(0.5, 6.5, 0.5):
            for angle in angles:
                tdx = dist * np.cos(angle)
                tdy = dist * np.sin(angle)
                if is_move_valid_fast(patch_union, tdx, tdy, r_union, o_union):
                    dx, dy = tdx, tdy
                    break
            if dx is not None:
                break

        if dx is not None:
            move_successful = True
            for i, geom in patch_geom_dict.items():
                init_area = poly_overlap_info[i]['overlap_area'] if i in poly_overlap_info else 0.0
                results[i] = (
                    shapely.affinity.translate(geom, xoff=dx, yoff=dy),
                    "moved",
                    init_area,
                    f"Shifted by {np.hypot(dx, dy):.2f}m"
                )

    # 2. Individual trimming fallback
    if not move_successful:
        patch_boundary = None
        if is_patch:
            try:
                patch_boundary = patch_union.boundary
            except Exception:
                patch_boundary = None

        for i in patch_indices:
            if i in poly_overlap_info:
                info = poly_overlap_info[i]
                geom = polys_geom_dict[i]
                init_area = info['overlap_area']

                eligible = False
                if not is_patch:
                    eligible = True
                else:
                    if patch_boundary is not None:
                        try:
                            if geom.boundary.intersects(patch_boundary):
                                eligible = True
                        except Exception:
                            pass

                if eligible and info['ratio'] < HIGH_OVERLAP_RATIO and enable_trim:
                    try:
                        trimmed = geom.difference(info['road_union'])
                        trimmed = extract_polygons(trimmed)

                        if trimmed and not trimmed.is_empty:
                            largest = largest_polygon(trimmed)
                            if is_shape_preserved(geom, largest):
                                results[i] = (largest, "edited", init_area, "Trimmed conflict area")
                            else:
                                results[i] = (geom, "unresolved", init_area, "Trimming violates shape preservation rules")
                        else:
                            results[i] = (geom, "unresolved", init_area, "Trim eliminated entire polygon")
                    except Exception as err:
                        results[i] = (geom, "unresolved", init_area, f"Geometry difference error: {err}")
                else:
                    reason = "Overlap > 20% threshold" if info['ratio'] >= HIGH_OVERLAP_RATIO else "Interior patch polygon"
                    results[i] = (geom, "unresolved", init_area, reason)
            else:
                results[i] = (polys_geom_dict[i], "unchanged", 0.0, "No conflict")

    return results, num_overlaps


# ── Full Processing & Reporting Engine ────────────────────────────────────────

def run_audit_process(
    road_input_path: str,
    poly_input_paths: list,
    output_excel_path: str,
    aoi_input_path: str = None,
    output_shapefile_path: str = None,
    enable_trim: bool = True,
    enable_delete: bool = True,
    enable_move: bool = True,
    progress_callback=None,
    log_callback=None
):
    temp_dir = tempfile.mkdtemp(prefix="gis_audit_")

    def log(m):
        if log_callback:
            log_callback(m)

    try:
        # 1. READ AND CONCATENATE ALL POLYGON INPUTS
        log("1. Reading & merging polygon datasets...")
        polys = load_multiple_polygons(poly_input_paths, temp_dir, log)
        log(f"   Total building/polygon records loaded: {len(polys):,}")

        orig_crs = polys.crs
        if polys.crs and polys.crs.is_geographic:
            log("   Projecting to local UTM meters for area computation...")
            utm_crs = polys.estimate_utm_crs()
            polys = polys.to_crs(utm_crs)
        else:
            utm_crs = polys.crs

        # Remove null or empty geometries
        polys = polys[polys.geometry.notna() & (~polys.geometry.is_empty)].reset_index(drop=True)

        # 2. READ AOI (IF PROVIDED)
        inside_aoi_mask = pd.Series(False, index=polys.index)
        if aoi_input_path and os.path.exists(aoi_input_path):
            log("2. Evaluating AOI bounding mask...")
            aoi_shp = resolve_shapefile(aoi_input_path, temp_dir)
            aoi_gdf = robust_load_shp(aoi_shp, log)
            if not aoi_gdf.empty:
                if aoi_gdf.crs != polys.crs:
                    aoi_gdf = aoi_gdf.to_crs(polys.crs)
                aoi_union = unary_union(aoi_gdf.geometry.values)
                if not aoi_union.is_valid:
                    aoi_union = shapely.make_valid(aoi_union)
                inside_aoi_mask = polys.geometry.intersects(aoi_union)
                log(f"   Protected inside AOI: {inside_aoi_mask.sum():,} records.")

        # 3. READ ROADS
        log("3. Reading road linework...")
        road_shp = resolve_shapefile(road_input_path, temp_dir)
        roads = robust_load_shp(road_shp, log)
        log(f"   Total road lines loaded: {len(roads):,}")

        if roads.crs != polys.crs:
            roads = roads.to_crs(polys.crs)
        roads = roads[roads.geometry.notna() & (~roads.geometry.is_empty)].reset_index(drop=True)

        # 4. PRE-SCREEN SMALL SLIVERS (<3 SQM) TOUCHING ROADS
        log("4. Screening sliver polygons touching roads...")
        has_bldg = pd.Series(False, index=polys.index)
        if 'BLDG_NAME' in polys.columns:
            has_bldg = polys['BLDG_NAME'].notna() & (polys['BLDG_NAME'].astype(str).str.strip() != '')

        protected_mask = has_bldg | inside_aoi_mask
        areas = polys.geometry.area
        to_delete = pd.Series(False, index=polys.index)

        road_geoms = roads.geometry.values
        road_tree = shapely.STRtree(road_geoms) if len(road_geoms) > 0 else None

        if enable_delete and road_tree is not None:
            small_mask = (areas < SMALL_AREA_THRESH) & (~protected_mask)
            if small_mask.any():
                s_indices = polys.index[small_mask].values
                s_geoms = polys.geometry.loc[s_indices].values
                hit_pos, _ = road_tree.query(s_geoms, predicate="intersects")
                if len(hit_pos) > 0:
                    del_idx = s_indices[np.unique(hit_pos)]
                    to_delete.loc[del_idx] = True
                    log(f"   Identified {len(del_idx):,} slivers (<3m²) scheduled for deletion.")

        movable_mask = ~(protected_mask | to_delete)
        movable_polys = polys[movable_mask]

        # 5. OVERLAP IDENTIFICATION
        log("5. Calculating initial polygon-to-road overlaps via spatial index...")
        movable_geoms = movable_polys.geometry.values
        mov_idx_map = movable_polys.index.values
        road_idx_map = roads.index.values

        p_pos, r_pos = road_tree.query(movable_geoms, predicate="intersects")

        poly_to_roads = defaultdict(list)
        intersecting_road_indices = set()
        for pp, rp in zip(p_pos, r_pos):
            p_orig = mov_idx_map[pp]
            r_orig = road_idx_map[rp]
            poly_to_roads[p_orig].append(r_orig)
            intersecting_road_indices.add(r_orig)

        log(f"   Found {len(poly_to_roads):,} polygons intersecting road network before correction.")

        # 6. TARGETED BUFFERING
        log(f"6. Buffering {len(intersecting_road_indices):,} intersecting road segments by {ROAD_BUFFER_M}m...")
        buffered_roads_dict = {}
        for r_idx in intersecting_road_indices:
            geom = roads.geometry.loc[r_idx]
            buffered_roads_dict[r_idx] = geom.buffer(ROAD_BUFFER_M, cap_style=2, join_style=2)

        patches = build_patches_fast(movable_polys)
        retained_polys = polys[~to_delete]
        retained_polys_geoms = retained_polys.geometry.values
        retained_idx_map = retained_polys.index.values
        retained_tree = shapely.STRtree(retained_polys_geoms)

        buffered_roads_geoms = np.array(list(buffered_roads_dict.values())) if buffered_roads_dict else np.array([])
        road_tree_buffered = shapely.STRtree(buffered_roads_geoms) if len(buffered_roads_geoms) > 0 else None

        # 7. PARALLEL CORRECTION EXECUTION
        log("7. Processing polygon corrections via parallel worker pool...")
        result_geoms = {}
        statuses = {}
        initial_overlap_areas = {}
        audit_reasons = {}

        for i in polys.index:
            if to_delete[i]:
                result_geoms[i] = None
                statuses[i] = "deleted"
                initial_overlap_areas[i] = polys.geometry.loc[i].area
                audit_reasons[i] = "Sliver polygon < 3 sq m touching road"
            elif protected_mask[i]:
                result_geoms[i] = polys.geometry.loc[i]
                statuses[i] = "unchanged"
                initial_overlap_areas[i] = 0.0
                audit_reasons[i] = "Protected (Inside AOI or has Building Name)"

        total_patches = len(patches)
        polys_geom_dict = polys.geometry.to_dict()
        max_workers = os.cpu_count() or 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_patch_task,
                    patch,
                    poly_to_roads,
                    polys_geom_dict,
                    buffered_roads_dict,
                    retained_polys_geoms,
                    retained_tree,
                    road_tree_buffered,
                    buffered_roads_geoms,
                    retained_idx_map,
                    enable_trim,
                    enable_move
                ): patch for patch in patches
            }

            done_count = 0
            for future in concurrent.futures.as_completed(futures):
                done_count += 1
                patch_res, _ = future.result()
                for idx, (res_geom, st, init_area, reason) in patch_res.items():
                    result_geoms[idx] = res_geom
                    statuses[idx] = st
                    initial_overlap_areas[idx] = init_area
                    audit_reasons[idx] = reason

                if progress_callback and (done_count % 100 == 0 or done_count == total_patches):
                    progress_callback(done_count, total_patches)

        # 8. PREPARE AUDIT DATASETS FOR EXCEL
        log("8. Generating coordinates & assembling Audit Excel artifacts...")

        # Compute centroid coordinates in EPSG:4326 (WGS84 Lat/Long)
        wgs84_polys = polys.to_crs(4326)
        rep_points = wgs84_polys.geometry.representative_point()
        lons = rep_points.x
        lats = rep_points.y

        # Build master analysis dataframe containing ALL original attributes
        audit_df = polys.drop(columns=["geometry"]).copy()
        audit_df.insert(0, "QC_STATUS", [statuses.get(i, "unchanged") for i in polys.index])
        audit_df.insert(1, "AUDIT_REASON", [audit_reasons.get(i, "") for i in polys.index])
        audit_df.insert(2, "INITIAL_OVERLAP_SQM", [round(initial_overlap_areas.get(i, 0.0), 3) for i in polys.index])
        audit_df.insert(3, "LATITUDE", lats)
        audit_df.insert(4, "LONGITUDE", lons)

        # Segregate into the exact artifacts required by Quality Management
        # Total Overlaps: Any building that had a genuine road conflict (Fixed + Unresolved)
        total_overlap_mask = audit_df["QC_STATUS"].isin(["edited", "moved", "unresolved"])
        df_total_overlaps = audit_df[total_overlap_mask].copy()

        # Fixed Using Automation: Trimmed or Moved
        automation_fixed_mask = audit_df["QC_STATUS"].isin(["edited", "moved"])
        df_fixed_automation = audit_df[automation_fixed_mask].copy()

        # Fixes Pending (Manual Review): Overlaps not resolved by automation
        unresolved_mask = audit_df["QC_STATUS"] == "unresolved"
        df_pending_manual = audit_df[unresolved_mask].copy()

        # Slivers Deleted
        deleted_mask = audit_df["QC_STATUS"] == "deleted"
        df_deleted = audit_df[deleted_mask].copy()

        count_total_overlap = len(df_total_overlaps)
        count_fixed_auto = len(df_fixed_automation)
        count_pending_man = len(df_pending_manual)
        count_deleted = len(df_deleted)

        log("\n" + "=" * 50)
        log(" AUDIT SUMMARY METRICS")
        log(f"  • Total Building Overlaps : {count_total_overlap:,}")
        log(f"  • Fixed Using Automation  : {count_fixed_auto:,}")
        log(f"  • Fixes Pending (Manual)  : {count_pending_man:,}")
        log(f"  • Deleted Slivers (<3m²)  : {count_deleted:,}")
        log("=" * 50 + "\n")

        # 9. WRITE MULTI-SHEET EXCEL
        log(f"9. Writing formatted Excel report to {os.path.basename(output_excel_path)}...")
        summary_kpis = pd.DataFrame([
            {"Metric": "Total Building Overlaps", "Count": count_total_overlap, "Notes": "Total initial conflicts detected"},
            {"Metric": "Fixed Using Automation", "Count": count_fixed_auto, "Notes": "Resolved via smart shift or precision trimming"},
            {"Metric": "Fixes Pending (Manual Review)", "Count": count_pending_man, "Notes": "Escalated for human operator review"},
            {"Metric": "Purged Slivers (<3 sq m)", "Count": count_deleted, "Notes": "Removed artifacts touching roads"},
            {"Metric": "Total Polygons Processed", "Count": len(polys), "Notes": "Total inputs across all uploaded files"}
        ])

        with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
            summary_kpis.to_excel(writer, sheet_name="Dashboard Summary", index=False)
            df_total_overlaps.to_excel(writer, sheet_name="Total Overlaps", index=False)
            df_fixed_automation.to_excel(writer, sheet_name="Fixed Using Automation", index=False)
            df_pending_manual.to_excel(writer, sheet_name="Fixes Pending (Manual)", index=False)
            if not df_deleted.empty:
                df_deleted.to_excel(writer, sheet_name="Deleted Slivers", index=False)

        # 10. OPTIONAL SHAPEFILE EXPORT
        if output_shapefile_path:
            log(f"10. Exporting corrected shapefile: {os.path.basename(output_shapefile_path)}...")
            export_mask = polys.index.map(lambda i: statuses.get(i) != "deleted")
            export_gdf = polys[export_mask].copy()
            export_gdf["geometry"] = export_gdf.index.map(result_geoms)
            export_gdf["STATUS"] = export_gdf.index.map(statuses)
            if orig_crs:
                export_gdf = export_gdf.to_crs(orig_crs)

            if HAS_PYOGRIO:
                export_gdf.to_file(output_shapefile_path, engine="pyogrio")
            else:
                export_gdf.to_file(output_shapefile_path, engine="fiona")

        log("Process completed successfully!")
        return {
            "total_overlaps": count_total_overlap,
            "fixed_auto": count_fixed_auto,
            "pending_manual": count_pending_man,
            "deleted": count_deleted
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Graphical User Interface (Tkinter) ────────────────────────────────────────

class MultiFileSelector(tk.Frame):
    """File selector capable of picking one or multiple .zip / .shp files."""
    def __init__(self, parent, label_text, **kw):
        super().__init__(parent, bg="#161B25", **kw)
        self.file_paths = []

        tk.Label(self, text=label_text, bg="#161B25", fg="#6B7A99",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        row = tk.Frame(self, bg="#161B25")
        row.pack(fill="x")

        self._text_var = tk.StringVar(value="No files selected")
        entry = tk.Entry(row, textvariable=self._text_var, bg="#1E2533", fg="#E8EDF5",
                         relief="flat", font=("Segoe UI", 10), state="readonly",
                         highlightbackground="#2A3347", highlightthickness=1)
        entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        btn = tk.Button(row, text="Add File(s)…", command=self._browse,
                        bg="#1E2533", fg="#4C8BF5", relief="flat", font=("Segoe UI", 9, "bold"),
                        padx=12, pady=6, cursor="hand2", highlightbackground="#2A3347", highlightthickness=1)
        btn.pack(side="right")

    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="Select Polygon Shapefiles or ZIP Archives",
            filetypes=[("GIS Datasets", "*.zip;*.shp"), ("ZIP Archives", "*.zip"), ("Shapefiles", "*.shp")]
        )
        if paths:
            self.file_paths = list(paths)
            if len(self.file_paths) == 1:
                self._text_var.set(os.path.basename(self.file_paths[0]))
            else:
                self._text_var.set(f"{len(self.file_paths)} files selected")

    def get_paths(self):
        return self.file_paths


class SingleFileSelector(tk.Frame):
    """Single file selector for Roads or AOI."""
    def __init__(self, parent, label_text, file_filter, **kw):
        super().__init__(parent, bg="#161B25", **kw)
        self.file_filter = file_filter

        tk.Label(self, text=label_text, bg="#161B25", fg="#6B7A99",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        row = tk.Frame(self, bg="#161B25")
        row.pack(fill="x")

        self._path_var = tk.StringVar()
        entry = tk.Entry(row, textvariable=self._path_var, bg="#1E2533", fg="#E8EDF5",
                         relief="flat", font=("Segoe UI", 10),
                         highlightbackground="#2A3347", highlightthickness=1)
        entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        btn = tk.Button(row, text="Browse…", command=self._browse,
                        bg="#1E2533", fg="#4C8BF5", relief="flat", font=("Segoe UI", 9, "bold"),
                        padx=12, pady=6, cursor="hand2", highlightbackground="#2A3347", highlightthickness=1)
        btn.pack(side="right")

    def _browse(self):
        path = filedialog.askopenfilename(title="Select File", filetypes=self.file_filter)
        if path:
            self._path_var.set(path)

    @property
    def path(self):
        return self._path_var.get().strip()


class AuditApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Building Overlap QC Audit & Auto-Correction Tool")
        self.geometry("960x860")
        self.configure(bg="#0F1117")
        self.minsize(860, 760)

        self._build_layout()

    def _build_layout(self):
        # Header
        header = tk.Frame(self, bg="#161B25", height=50)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="⬡ BUILDING-ROAD QC AUDIT & EXCEL GENERATOR",
                 bg="#161B25", fg="#4C8BF5", font=("Segoe UI", 12, "bold")).pack(side="left", padx=20, pady=12)

        # Body
        body = tk.Frame(self, bg="#0F1117")
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Controls panel (Left)
        left = tk.Frame(body, bg="#161B25", highlightbackground="#2A3347", highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # KPI panel (Right)
        right = tk.Frame(body, bg="#161B25", width=290, highlightbackground="#2A3347", highlightthickness=1)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._setup_inputs(left)
        self._setup_metrics(right)

    def _setup_inputs(self, parent):
        pad = {"padx": 16, "pady": 4}

        tk.Label(parent, text="INPUT DATASETS (.ZIP OR .SHP)", bg="#161B25", fg="#6B7A99",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(10, 2))

        self.poly_selector = MultiFileSelector(parent, "Building Polygon Datasets (Multiple Files Allowed)")
        self.poly_selector.pack(fill="x", **pad)

        self.road_selector = SingleFileSelector(
            parent, "Road Network (.zip or .shp)",
            [("Road Files", "*.zip;*.shp"), ("ZIP Archives", "*.zip"), ("Shapefiles", "*.shp")]
        )
        self.road_selector.pack(fill="x", **pad)

        self.aoi_selector = SingleFileSelector(
            parent, "Optional AOI Boundary (.zip or .shp)",
            [("AOI Files", "*.zip;*.shp"), ("All files", "*.*")]
        )
        self.aoi_selector.pack(fill="x", **pad)

        # Output Excel
        tk.Label(parent, text="EXCEL AUDIT ARTIFACTS", bg="#161B25", fg="#6B7A99",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(10, 2))

        row_out = tk.Frame(parent, bg="#161B25")
        row_out.pack(fill="x", **pad)

        self.out_excel_var = tk.StringVar()
        tk.Entry(row_out, textvariable=self.out_excel_var, bg="#1E2533", fg="#E8EDF5",
                 relief="flat", font=("Segoe UI", 10),
                 highlightbackground="#2A3347", highlightthickness=1).pack(
                     side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        tk.Button(row_out, text="Save Excel As…", command=self._choose_excel_out,
                  bg="#1E2533", fg="#4C8BF5", relief="flat", font=("Segoe UI", 9, "bold"),
                  padx=12, pady=6, cursor="hand2", highlightbackground="#2A3347", highlightthickness=1).pack(side="right")

        # Progress bar
        tk.Label(parent, text="PROGRESS", bg="#161B25", fg="#6B7A99",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(10, 2))

        self.progress_lbl = tk.Label(parent, text="Ready", bg="#161B25", fg="#6B7A99", font=("Segoe UI", 9))
        self.progress_lbl.pack(anchor="w", padx=16, pady=(0, 2))

        self.progress_bar = ttk.Progressbar(parent, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 10))

        # Log
        tk.Label(parent, text="EXECUTION LOG", bg="#161B25", fg="#6B7A99",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(0, 2))

        log_box = tk.Frame(parent, bg="#1E2533", highlightbackground="#2A3347", highlightthickness=1)
        log_box.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        self.log_text = tk.Text(log_box, bg="#1E2533", fg="#E8EDF5", font=("Courier New", 9),
                                relief="flat", wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_box, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

        # Run Button
        self.run_btn = tk.Button(parent, text="▶  GENERATE QC AUDIT & EXCEL", command=self._start,
                                 bg="#4C8BF5", fg="#FFFFFF", relief="flat", font=("Segoe UI", 11, "bold"),
                                 cursor="hand2", pady=10)
        self.run_btn.pack(fill="x", padx=16, pady=(0, 14))

    def _setup_metrics(self, parent):
        tk.Label(parent, text="AUDIT METRICS", bg="#161B25", fg="#6B7A99",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(16, 12))

        self.card_total = self._create_card(parent, "Total Building Overlaps", "#4C8BF5")
        self.card_fixed = self._create_card(parent, "Fixed Using Automation", "#34D399")
        self.card_manual = self._create_card(parent, "Fixes Pending (Manual)", "#FBBF24")
        self.card_deleted = self._create_card(parent, "Purged Slivers (<3m²)", "#F87171")

    def _create_card(self, parent, title, color):
        frame = tk.Frame(parent, bg="#1E2533", highlightbackground=color, highlightthickness=1)
        frame.pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(frame, text=title, bg="#1E2533", fg="#6B7A99",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=12, pady=(8, 0))
        var = tk.StringVar(value="0")
        tk.Label(frame, textvariable=var, bg="#1E2533", fg=color,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=12, pady=(2, 8))
        return var

    def _choose_excel_out(self):
        p = filedialog.asksaveasfilename(title="Save Audit Excel", defaultextension=".xlsx",
                                         filetypes=[("Excel Workbook", "*.xlsx")])
        if p:
            self.out_excel_var.set(p)

    def log(self, text):
        def _append():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(0, _append)

    def _start(self):
        poly_paths = self.poly_selector.get_paths()
        road_path = self.road_selector.path
        aoi_path = self.aoi_selector.path
        excel_out = self.out_excel_var.get().strip()

        if not poly_paths:
            messagebox.showerror("Missing Input", "Please select at least one Polygon .zip or .shp file.")
            return
        if not road_path:
            messagebox.showerror("Missing Input", "Please select a Road .zip or .shp file.")
            return
        if not excel_out:
            messagebox.showerror("Missing Output", "Please designate a destination path for the Excel workbook.")
            return

        self.run_btn.configure(state="disabled", text="⏳ Running Analysis…", bg="#2A3347")
        self.progress_bar["value"] = 0

        def runner():
            try:
                def update_prog(curr, tot):
                    pct = int(curr / tot * 100) if tot else 0
                    self.after(0, lambda: self.progress_bar.configure(value=pct))
                    self.after(0, lambda: self.progress_lbl.configure(text=f"Evaluating component {curr:,}/{tot:,} ({pct}%)"))

                res = run_audit_process(
                    road_input_path=road_path,
                    poly_input_paths=poly_paths,
                    output_excel_path=excel_out,
                    aoi_input_path=aoi_path if aoi_path else None,
                    progress_callback=update_prog,
                    log_callback=self.log
                )
                self.after(0, lambda: self._on_success(res, excel_out))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._on_failure(str(e)))

        import threading
        threading.Thread(target=runner, daemon=True).start()

    def _on_success(self, res, path):
        self.run_btn.configure(state="normal", text="▶  GENERATE QC AUDIT & EXCEL", bg="#4C8BF5")
        self.progress_bar["value"] = 100
        self.progress_lbl.configure(text="Audit Completed ✓", fg="#34D399")

        self.card_total.set(f"{res['total_overlaps']:,}")
        self.card_fixed.set(f"{res['fixed_auto']:,}")
        self.card_manual.set(f"{res['pending_manual']:,}")
        self.card_deleted.set(f"{res['deleted']:,}")

        messagebox.showinfo("Audit Finished", f"Quality report successfully saved:\n{path}")

    def _on_failure(self, err):
        self.run_btn.configure(state="normal", text="▶  GENERATE QC AUDIT & EXCEL", bg="#4C8BF5")
        self.progress_lbl.configure(text="Execution Failed ✗", fg="#F87171")
        self.log(f"\n[ERROR]: {err}")
        messagebox.showerror("Execution Error", err)


if __name__ == "__main__":
    app = AuditApp()
    app.mainloop()