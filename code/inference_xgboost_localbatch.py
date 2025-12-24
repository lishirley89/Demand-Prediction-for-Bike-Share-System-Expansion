#!/usr/bin/env python3
"""
use python environment python3.13.5 /usr/local/bin/python3 --> python3
Inference script: Build features for a candidate station (lat/lng) and predict demand.

Pipeline per request (single point):
- Generate 12 rows (one per month)
- Compute distance-band station density (n250..n1500) from historical stations
- Create 500m buffer for the point
- Compute infrastructure metrics within buffer (bike routes length, streets length, rail/bus stop counts)
- Compute POI counts via ArcGIS (tourism/education/medical/shops/leisure), with timeouts and fallback to zeros
- Spatially join to census tract and extract demographic/economic variables
- month_sin, month_cos
- Align features to training schema (columns of training_dataset_preprocessed.csv minus targets/ids)
- Load XGBoost models and return predictions for cbike/ebike start/end

"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

import geopandas as gpd
from shapely.geometry import Point

# Local configuration
# Paths are relative to project root when script is run from there
DATA_PREFIX = Path("data")
MODEL_PREFIX = Path("result")

# File names
STATIONS_CSV = "master_stations.csv"
CENSUS_TRACTS_GEOJSON = "cook_county_census_tracts.geojson"
BIKE_ROUTES_GEOJSON = "Bike_Routes_20250828.geojson"
RAIL_STATIONS_GEOJSON = "CTA_-_'L'_(Rail)_Stations_20250828.geojson"
BUS_STOPS_GEOJSON = "CTA_BusStops_20250828.geojson"
STREET_CENTERLINES_GEOJSON = "transportation_streetcenterlines20250828.geojson"
TRAIN_PREP_CSV = "training_dataset_preprocessed.csv"


# Trained models
MODEL_KEYS = {
    "cbike_start": "xgboost_model_cbike_start.joblib",
    "cbike_end": "xgboost_model_cbike_end.joblib",
    "ebike_start": "xgboost_model_ebike_start.joblib",
    "ebike_end": "xgboost_model_ebike_end.joblib",
}

# Training feature schema reference (use columns to align features)

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
    bike_routes_path = DATA_PREFIX / BIKE_ROUTES_GEOJSON
    street_centerlines_path = DATA_PREFIX / STREET_CENTERLINES_GEOJSON
    rail_stations_path = DATA_PREFIX / RAIL_STATIONS_GEOJSON
    bus_stops_path = DATA_PREFIX / BUS_STOPS_GEOJSON

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
    tracts_path = DATA_PREFIX / CENSUS_TRACTS_GEOJSON
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


def build_month_rows() -> List[Tuple[int, Dict[str, float]]]:
    """Return list of (month, month_cyclic_dict) for 1..12 with month_sin and month_cos."""
    rows: List[Tuple[int, Dict[str, float]]] = []
    for m in range(1, 13):
        # Cyclic encoding: sin(2π * month / 12) and cos(2π * month / 12)
        month_sin = math.sin(2 * math.pi * m / 12)
        month_cos = math.cos(2 * math.pi * m / 12)
        cyclic = {
            "month_sin": month_sin,
            "month_cos": month_cos
        }
        rows.append((m, cyclic))
    return rows


def align_to_training_schema(df_features: pd.DataFrame, model=None) -> Tuple[pd.DataFrame, List[str]]:
    """Align feature columns to the model's expected features: drop extras, add missing=0."""
    # Get expected features from the model if available, otherwise from CSV schema
    if model is not None and hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)
    else:
        # Fallback to CSV schema
        schema_path = MODEL_PREFIX / TRAIN_PREP_CSV
        train = pd.read_csv(schema_path, nrows=100)
        expected_cols = [c for c in train.columns if c not in TARGET_COLS + ID_COLS]

    # Add missing columns with 0, drop unknowns
    for c in expected_cols:
        if c not in df_features.columns:
            df_features[c] = 0
    df_aligned = df_features[expected_cols].copy()
    return df_aligned, expected_cols


def load_models() -> Dict[str, object]:
    models: Dict[str, object] = {}
    for target, filename in MODEL_KEYS.items():
        model_path = MODEL_PREFIX / filename
        models[target] = joblib.load(model_path)
    return models


def predict_for_point(lat: float, lng: float, models: Dict[str, object] = None, stations_df: pd.DataFrame = None) -> pd.DataFrame:
    """Predict demand for a single point. Models and stations_df can be pre-loaded for batch processing."""
    # Load reference data if not provided
    if stations_df is None:
        stations_path = MODEL_PREFIX / STATIONS_CSV
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
    for month, month_cyclic in build_month_rows():
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
        }
        base.update(month_cyclic)  # Adds month_sin and month_cos
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

    # Predict (load models if not provided)
    if models is None:
        models = load_models()
    
    # Align to training schema using the first model's expected features
    # (all models should have the same feature schema)
    first_model = list(models.values())[0] if models else None
    X, feature_cols = align_to_training_schema(df, model=first_model)
    
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
    # Month is already in the rows from build_month_rows()
    out.insert(2, "month", [m for m, _ in build_month_rows()])

    return out


def main():
    """Batch inference for all H3 cells in Chicago."""
    parser = argparse.ArgumentParser(
        description="Batch predict bike share demand for all H3 cells in Chicago",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_xgboost_localbatch.py
  python inference_xgboost_localbatch.py --input result/chicago_h3_res8.csv --output result/chicago_predictions_res8.csv
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        default="result/chicago_h3_res8.csv",
        help="Input CSV file with H3 cells (default: result/chicago_h3_res8.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result/chicago_h3_predictions_res8.csv",
        help="Output CSV file path (default: result/chicago_h3_predictions_res8.csv)"
    )

    args = parser.parse_args()

    try:
        # Load H3 cells
        print(f"Loading H3 cells from {args.input}...")
        h3_df = pd.read_csv(args.input)
        print(f"Found {len(h3_df)} H3 cells to process")

        # Load models once (reused for all predictions)
        print("Loading models...")
        models = load_models()
        print("Models loaded successfully")

        # Load reference data once (reused for all predictions)
        print("Loading reference data...")
        stations_path = MODEL_PREFIX / STATIONS_CSV
        stations_df = pd.read_csv(stations_path)
        print("Reference data loaded")

        # Process each H3 cell
        all_results = []
        total = len(h3_df)

        for idx, row in h3_df.iterrows():
            print(f"Processing {idx + 1}/{total}: H3 {row['h3_index']}")
            h3_index = row['h3_index']
            lat = row['lat']
            lng = row['lng']

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"Processing {idx + 1}/{total}: H3 {h3_index}")

            try:
                # Get predictions for this location (pass pre-loaded models and stations_df)
                df_pred = predict_for_point(lat, lng, models=models, stations_df=stations_df)

                # Add h3_index to each row
                df_pred['h3_index'] = h3_index

                # Select only the columns we need: h3_index, lat, lng, month, and predictions
                result_cols = ['h3_index', 'lat', 'lng', 'month', 'cbike_start', 'cbike_end', 'ebike_start', 'ebike_end']
                df_result = df_pred[result_cols].copy()

                all_results.append(df_result)

            except Exception as e:
                print(f"Error processing H3 {h3_index} ({lat}, {lng}): {e}", file=sys.stderr)
                continue

        # Combine all results
        print(f"\nCombining results from {len(all_results)} locations...")
        final_df = pd.concat(all_results, ignore_index=True)

        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Saved {len(final_df)} predictions to {output_path}")
        print(f"Total locations: {len(all_results)}")
        print(f"Total predictions (12 months × locations): {len(final_df)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
