#!/usr/bin/env python3
"""
Inference script: Build features for a candidate station (lat/lng) and predict demand.

Pipeline per request (single point):
- Generate 12 rows (one per month)
- Compute distance-band station density (n250..n1500) from historical stations
- Create 500m buffer for the point
- Compute infrastructure metrics within buffer (bike routes length, streets length, rail/bus stop counts)
- Compute POI counts via ArcGIS (tourism/education/medical/shops/leisure), with timeouts and fallback to zeros
- Spatially join to census tract and extract demographic/economic variables
- One-hot encode month (month_1..month_12) consistent with training
- Align features to training schema (columns of training_dataset_preprocessed.csv minus targets/ids)
- Load XGBoost models and return predictions for cbike/ebike start/end

Usage:
  python code/inference.py --lat 41.88 --lng -87.63 --year 2025 --out result/inference_sample.csv

Note:
- Requires the data files present locally under data/ and result/ as produced by the repo pipeline
- For ArcGIS POI calls, network connectivity is required; failures will fallback to zeros
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

# Project paths (local defaults)
PROJECT_ROOT = Path("/Users/shirley/Documents/Research/demand/parkchargebike")
DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "result"

# Optional S3 configuration via environment variables
DATA_S3_PREFIX = os.getenv("DATA_S3_PREFIX", "")  # e.g., s3://lishirley89/data
MODELS_S3_PREFIX = os.getenv("MODELS_S3_PREFIX", "")  # e.g., s3://lishirley89/models/xgboost/v1

# Select a writable cache directory. Prefer env, fallback to /tmp.
def _init_cache_dir() -> Path:
    prefer = os.getenv("CACHE_DIR", "/tmp")
    p = Path(prefer)
    try:
        p.mkdir(parents=True, exist_ok=True)
        # attempt a tiny write test to ensure writable
        test_file = p / ".write_test"
        test_file.write_text("ok")
        test_file.unlink(missing_ok=True)
        return p
    except Exception:
        # last resort, use /tmp
        fallback = Path("/tmp")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

CACHE_DIR = _init_cache_dir()

# Reference input files (local defaults; may be overridden by S3)
STATIONS_CSV = RESULT_DIR / "master_stations.csv"  # station_id, station_name, lat, lng, ...
CENSUS_TRACTS_GEOJSON = DATA_DIR / "cook_county_census_tracts.geojson"
BIKE_ROUTES_GEOJSON = DATA_DIR / "Bike_Routes_20250828.geojson"
RAIL_STATIONS_GEOJSON = DATA_DIR / "CTA_-_'L'_(Rail)_Stations_20250828.geojson"
BUS_STOPS_GEOJSON = DATA_DIR / "CTA_BusStops_20250828.geojson"
STREET_CENTERLINES_GEOJSON = DATA_DIR / "transportation_streetcenterlines20250828.geojson"

def _s3_to_local(s3_uri: str, cache_dir: Path) -> Path:
    """Download s3://bucket/key to cache_dir, returning local Path. If already exists, reuse."""
    try:
        import boto3  # type: ignore
    except Exception as e:
        raise RuntimeError("boto3 is required to fetch S3 artifacts. Install it or bake into image.") from e

    assert s3_uri.startswith("s3://"), f"Not an S3 URI: {s3_uri}"
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    local_path = cache_dir / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))
    return local_path


def _resolve_path(local_path: Path, s3_prefix: str, rel_key: str) -> Path:
    """Return a local path, downloading from S3 if s3_prefix is provided (s3://...)."""
    if s3_prefix and s3_prefix.startswith("s3://"):
        s3_uri = s3_prefix.rstrip("/") + "/" + rel_key.lstrip("/")
        return _s3_to_local(s3_uri, CACHE_DIR)
    return local_path


# Trained models (resolved at load time with S3 support)
MODEL_KEYS = {
    "cbike_start": "xgboost_model_cbike_start.joblib",
    "cbike_end": "xgboost_model_cbike_end.joblib",
    "ebike_start": "xgboost_model_ebike_start.joblib",
    "ebike_end": "xgboost_model_ebike_end.joblib",
}

# Training feature schema reference (use columns to align features)
TRAIN_PREP_CSV = RESULT_DIR / "training_dataset_preprocessed.csv"

# Targets and ID columns (to exclude when aligning features)
TARGET_COLS = [
    "cbike_start", "cbike_end", "ebike_start", "ebike_end", "total_start", "total_end",
]
ID_COLS = ["station_id", "year"]

# Distance bands in meters
DISTANCE_BANDS_M = [250, 500, 750, 1000, 1250, 1500]

# ArcGIS REST service URLs for POI categories
ARCGIS_SERVICES: Dict[str, str] = {
    "poi_tourism": "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Tourism/FeatureServer/0/query",
    "poi_education": "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Educational/FeatureServer/0/query",
    "poi_medical": "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Medical/FeatureServer/0/query",
    "poi_shop": "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Shops/FeatureServer/0/query",
    "poi_leisure": "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Leisure/FeatureServer/0/query",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometers."""
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_distance_bands(lat: float, lng: float, stations_df: pd.DataFrame) -> Dict[str, int]:
    """Count number of stations within specified distance bands from the point."""
    # Ensure numeric
    stations_df = stations_df.copy()
    stations_df["lat"] = pd.to_numeric(stations_df["lat"], errors="coerce")
    stations_df["lng"] = pd.to_numeric(stations_df["lng"], errors="coerce")
    stations_df = stations_df.dropna(subset=["lat", "lng"]).reset_index(drop=True)

    distances_m = []
    for _, row in stations_df.iterrows():
        d_km = haversine_km(lat, lng, float(row["lat"]), float(row["lng"]))
        distances_m.append(d_km * 1000.0)

    distances_m = np.array(distances_m)

    # Bands: 0-250, 250-500, 500-750, 750-1000, 1000-1250, 1250-1500
    counts = {}
    prev = 0
    for band in DISTANCE_BANDS_M:
        mask = (distances_m > prev) & (distances_m <= band)
        counts[f"n{band}"] = int(mask.sum())
        prev = band
    return counts


def build_buffer_gdf(lat: float, lng: float, radius_m: float = 500.0) -> gpd.GeoDataFrame:
    """Create a 500m buffer around the point in a metric CRS for area/length ops."""
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(lng, lat)], crs="EPSG:4326")
    # Project to metric CRS for Chicago area; use EPSG:3857 general web mercator
    gdf_m = gdf.to_crs("EPSG:3857")
    buffer_geom = gdf_m.geometry.buffer(radius_m).iloc[0]
    buffer_gdf_m = gpd.GeoDataFrame({"id": [1]}, geometry=[buffer_geom], crs="EPSG:3857")
    return buffer_gdf_m


def length_of_lines_within_buffer(lines_gdf: gpd.GeoDataFrame, buffer_gdf_m: gpd.GeoDataFrame) -> float:
    if lines_gdf.empty:
        return 0.0
    lines_m = lines_gdf.to_crs(buffer_gdf_m.crs)
    inter = gpd.overlay(lines_m, buffer_gdf_m, how="intersection")
    if inter.empty:
        return 0.0
    return float(inter.length.sum())


def count_points_within_buffer(points_gdf: gpd.GeoDataFrame, buffer_gdf_m: gpd.GeoDataFrame) -> int:
    if points_gdf.empty:
        return 0
    points_m = points_gdf.to_crs(buffer_gdf_m.crs)
    within_mask = points_m.within(buffer_gdf_m.geometry.iloc[0])
    return int(within_mask.sum())


def load_geodataframes() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load bike routes (lines), street centerlines (lines), rail stations (points), bus stops (points)."""
    bike_routes_path = _resolve_path(BIKE_ROUTES_GEOJSON, DATA_S3_PREFIX, BIKE_ROUTES_GEOJSON.name)
    street_centerlines_path = _resolve_path(STREET_CENTERLINES_GEOJSON, DATA_S3_PREFIX, STREET_CENTERLINES_GEOJSON.name)
    rail_stations_path = _resolve_path(RAIL_STATIONS_GEOJSON, DATA_S3_PREFIX, RAIL_STATIONS_GEOJSON.name)
    bus_stops_path = _resolve_path(BUS_STOPS_GEOJSON, DATA_S3_PREFIX, BUS_STOPS_GEOJSON.name)

    bike_routes = gpd.read_file(bike_routes_path)
    street_centerlines = gpd.read_file(street_centerlines_path)
    rail_stations = gpd.read_file(rail_stations_path)
    bus_stops = gpd.read_file(bus_stops_path)

    # Ensure CRS
    for g in [bike_routes, street_centerlines, rail_stations, bus_stops]:
        if g.crs is None:
            g.set_crs("EPSG:4326", inplace=True)
    return bike_routes, street_centerlines, rail_stations, bus_stops


def arcgis_count_points_in_buffer(lat: float, lng: float, radius_m: float, timeout_s: float = 10.0) -> Dict[str, int]:
    """Query ArcGIS feature services for POI counts within a circular buffer.

    Implementation: approximate circle as bounding box for speed and to avoid geometry encoding.
    We slightly expand the bbox (~5%) to be conservative. This returns an upper bound; acceptable for inference.
    """
    # Approximate bbox in WGS84 from meter radius using ~111,320 m per degree at equator
    # For Chicago lat, adjust by cos(lat)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lng = 111_320.0 * math.cos(math.radians(lat))
    dlat = (radius_m / meters_per_deg_lat) * 1.05
    dlng = (radius_m / meters_per_deg_lng) * 1.05

    minx, miny = (lng - dlng), (lat - dlat)
    maxx, maxy = (lng + dlng), (lat + dlat)

    params_base = {
        "f": "json",
        "where": "1=1",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID",
        "returnGeometry": False,
        "returnCountOnly": True,
    }

    counts: Dict[str, int] = {k: 0 for k in ARCGIS_SERVICES.keys()}

    for key, url in ARCGIS_SERVICES.items():
        params = params_base.copy()
        params["geometry"] = json.dumps({
            "xmin": float(minx), "ymin": float(miny), "xmax": float(maxx), "ymax": float(maxy), "spatialReference": {"wkid": 4326}
        })
        try:
            r = requests.get(url, params=params, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            cnt = int(data.get("count", 0))
            counts[key] = cnt
        except Exception:
            # Leave default 0 on failures
            counts[key] = counts.get(key, 0)
    return counts


def join_census(lat: float, lng: float) -> Dict[str, float]:
    """Spatially join point to census tract and extract variables used in training (if present)."""
    gdf_pt = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(lng, lat)], crs="EPSG:4326")
    tracts_path = _resolve_path(CENSUS_TRACTS_GEOJSON, DATA_S3_PREFIX, CENSUS_TRACTS_GEOJSON.name)
    tracts = gpd.read_file(tracts_path)
    if tracts.crs is None:
        tracts.set_crs("EPSG:4269", inplace=True)
    tracts = tracts.to_crs("EPSG:4326")

    joined = gpd.sjoin(gdf_pt, tracts, how="left", predicate="within")
    if joined.empty:
        return {}

    row = joined.iloc[0]
    # Known variables from calc_station_census.py CENSUS_VALUE_COLS
    candidate_cols = [
        "total_population", "pct_white", "pct_black", "pct_asian", "pct_indian", "pct_hawaiian",
        "pct_two_or_more_races", "pct_hispanic", "pct_female", "pct_young_adults_20_34",
        "pct_zero_car_ownership", "unemployment_rate", "pct_bachelors_plus", "pct_drive_alone",
        "pct_bike_to_work", "pct_walk_to_work", "housing_density", "per_capita_income",
        "land_area_sq_meters", "population_density_sq_meter", "housing_density_sq_meter",
    ]
    values = {}
    for c in candidate_cols:
        if c in joined.columns:
            try:
                values[c] = float(row[c]) if pd.notna(row[c]) else 0.0
            except Exception:
                values[c] = 0.0
    return values


@dataclass
class InfraMetrics:
    bike_route_length_m: float
    street_length_m: float
    rail_stops_count: int
    bus_stops_count: int


def compute_infrastructure_metrics(lat: float, lng: float, buffer_radius_m: float = 500.0) -> InfraMetrics:
    buffer_gdf_m = build_buffer_gdf(lat, lng, buffer_radius_m)
    bike_routes, street_centerlines, rail_stations, bus_stops = load_geodataframes()

    bike_len_m = length_of_lines_within_buffer(bike_routes, buffer_gdf_m)
    streets_len_m = length_of_lines_within_buffer(street_centerlines, buffer_gdf_m)
    rail_cnt = count_points_within_buffer(rail_stations, buffer_gdf_m)
    bus_cnt = count_points_within_buffer(bus_stops, buffer_gdf_m)

    return InfraMetrics(
        bike_route_length_m=float(bike_len_m),
        street_length_m=float(streets_len_m),
        rail_stops_count=int(rail_cnt),
        bus_stops_count=int(bus_cnt),
    )


def build_month_rows(year: int) -> List[Tuple[int, Dict[str, int]]]:
    """Return list of (month, month_one_hot_dict) for 1..12."""
    rows: List[Tuple[int, Dict[str, int]]] = []
    for m in range(1, 13):
        one_hot = {f"month_{i}": 1 if i == m else 0 for i in range(1, 13)}
        rows.append((m, one_hot))
    return rows


def align_to_training_schema(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Align feature columns to the training preprocessed schema: drop extras, add missing=0."""
    try:
        schema_path = _resolve_path(TRAIN_PREP_CSV, DATA_S3_PREFIX, "data/training_dataset_preprocessed.csv")
        train = pd.read_csv(schema_path, nrows=100)
    except Exception as e:
        print(f"Warning: Could not load schema from S3, using local file: {e}")
        train = pd.read_csv(TRAIN_PREP_CSV, nrows=100)
    expected_cols = [c for c in train.columns if c not in TARGET_COLS + ID_COLS]

    # Add missing columns with 0, drop unknowns
    for c in expected_cols:
        if c not in df_features.columns:
            df_features[c] = 0
    df_aligned = df_features[expected_cols].copy()
    return df_aligned, expected_cols


def load_models() -> Dict[str, object]:
    models: Dict[str, object] = {}
    for target, rel_key in MODEL_KEYS.items():
        # Resolve model path from S3 if configured, else use local result/
        local_default = RESULT_DIR / rel_key
        model_path = _resolve_path(local_default, MODELS_S3_PREFIX, rel_key)
        if not model_path.exists():
            print(f"Warning: model not found for {target}: {model_path}")
            continue
        models[target] = joblib.load(model_path)
    return models


def predict_for_point(lat: float, lng: float, year: int) -> pd.DataFrame:
    # Load reference data
    stations_path = _resolve_path(STATIONS_CSV, DATA_S3_PREFIX, "master_stations.csv")
    stations_df = pd.read_csv(stations_path)

    # Compute static features independent of month
    band_counts = compute_distance_bands(lat, lng, stations_df)
    infra = compute_infrastructure_metrics(lat, lng, 500.0)

    # POIs via ArcGIS (with fallback)
    poi_counts = arcgis_count_points_in_buffer(lat, lng, 500.0)

    # Census join
    census_vals = join_census(lat, lng)

    # Build 12 rows
    rows = []
    for month, month_one_hot in build_month_rows(year):
        base = {
            # distance bands
            "n250": band_counts.get("n250", 0),
            "n500": band_counts.get("n500", 0),
            "n750": band_counts.get("n750", 0),
            "n1000": band_counts.get("n1000", 0),
            "n1250": band_counts.get("n1250", 0),
            "n1500": band_counts.get("n1500", 0),
            # infrastructure metrics
            "bike_route_length_m": infra.bike_route_length_m,
            "street_length_m": infra.street_length_m,
            "rail_stops_count": infra.rail_stops_count,
            "bus_stops_count": infra.bus_stops_count,
            # POIs
            "poi_tourism": poi_counts.get("poi_tourism", 0),
            "poi_education": poi_counts.get("poi_education", 0),
            "poi_medical": poi_counts.get("poi_medical", 0),
            "poi_shop": poi_counts.get("poi_shop", 0),
            "poi_leisure": poi_counts.get("poi_leisure", 0),
            # identifiers
            "year": year,
        }
        base.update(month_one_hot)
        base.update(census_vals)
        rows.append(base)

    df = pd.DataFrame(rows)

    # Add log transforms where training had them (best-effort): create log_* for known fields
    def safe_log(x: pd.Series) -> pd.Series:
        return np.log(np.clip(x.astype(float), a_min=1e-9, a_max=None))

    for col in [
        "n250", "n500", "n750", "n1000", "n1250", "n1500",
        "rail_stops_count", "poi_tourism", "poi_education", "poi_medical", "poi_shop",
        "population_density_sq_meter", "pct_black", "pct_asian", "pct_indian", "pct_hawaiian",
        "pct_two_or_more_races", "pct_hispanic", "pct_female", "unemployment_rate",
        "pct_bike_to_work", "pct_walk_to_work", "bike_route_length_m", "street_length_m",
    ]:
        if col in df.columns:
            df[f"log_{col}"] = safe_log(df[col])

    # Align to training schema
    X, feature_cols = align_to_training_schema(df)

    # Predict
    models = load_models()
    preds: Dict[str, np.ndarray] = {}
    for target, model in models.items():
        try:
            preds[target] = model.predict(X)
        except Exception as e:
            print(f"Prediction failed for {target}: {e}")

    out = df.copy()
    for target, arr in preds.items():
        out[target] = arr

    # Add lat/lng and month for clarity
    out.insert(0, "lat", lat)
    out.insert(1, "lng", lng)
    # Derive month number from month_*
    out.insert(2, "month", [int(np.argmax([row.get(f"month_{i}", 0) for i in range(1, 13)]) + 1) for _, row in df.iterrows()])

    return out


def main():
    parser = argparse.ArgumentParser(description="Predict demand for a candidate station (lat/lng)")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lng", type=float, required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", type=str, default=str(RESULT_DIR / "inference_output.csv"))
    args = parser.parse_args()

    if not STATIONS_CSV.exists():
        print(f"Error: stations file not found: {STATIONS_CSV}")
        sys.exit(1)
    if not TRAIN_PREP_CSV.exists():
        print(f"Error: training preprocessed file not found: {TRAIN_PREP_CSV}")
        sys.exit(1)

    df_out = predict_for_point(args.lat, args.lng, args.year)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()
