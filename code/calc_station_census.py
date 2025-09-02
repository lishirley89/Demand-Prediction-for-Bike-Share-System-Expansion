#!/usr/bin/env python3
"""
Create station-to-census mapping by spatially joining stations to tracts.

- Loads Cook County census tracts GeoJSON with key variables
- Loads master_stations.csv (expects columns: station_id, lat, lng)
- Converts stations to points and performs spatial join (point-in-polygon)
- Writes result/master_stations_census.csv with appended census variables
"""

import os
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

TRACTS_PATH = "data/cook_county_census_tracts.geojson"
STATIONS_PATH = "result/master_stations.csv"
OUTPUT_PATH = "result/master_stations_census.csv"

# Census key variables we expect (filter to what's present to be safe)
CENSUS_ID_COLS = [
    "GEOID",
    "TRACTCE",
    "NAME",
]
CENSUS_VALUE_COLS = [
    "total_population",
    "pct_white",
    "pct_black",
    "pct_asian",
    "pct_indian",
    "pct_hawaiian",
    "pct_two_or_more_races",
    "pct_hispanic",
    "pct_female",
    "pct_young_adults_20_34",
    "pct_zero_car_ownership",
    "unemployment_rate",
    "pct_bachelors_plus",
    "pct_drive_alone",
    "pct_bike_to_work",
    "pct_walk_to_work",
    "housing_density",
    "per_capita_income",
    "land_area_sq_meters",
    "population_density_sq_meter",
    "housing_density_sq_meter",
]


def load_census_tracts(tracts_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(tracts_path):
        print(f"Error: Tracts file not found at {tracts_path}")
        sys.exit(1)
    gdf = gpd.read_file(tracts_path)
    if gdf.crs is None:
        # Fallback if CRS missing; GeoJSON from earlier pipeline used EPSG:4269
        gdf.set_crs("EPSG:4269", inplace=True)
    return gdf


def load_stations(stations_path: str) -> pd.DataFrame:
    if not os.path.exists(stations_path):
        print(f"Error: Stations file not found at {stations_path}")
        sys.exit(1)
    df = pd.read_csv(stations_path)
    # Validate columns
    for col in ["station_id", "lat", "lng"]:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing in {stations_path}")
            sys.exit(1)
    return df


def convert_stations_to_gdf(stations_df: pd.DataFrame, target_crs) -> gpd.GeoDataFrame:
    # Stations lat/lng assumed WGS84 (EPSG:4326)
    geometry = [Point(lon, lat) for lat, lon in zip(stations_df["lat"], stations_df["lng"])]
    stations_gdf = gpd.GeoDataFrame(stations_df.copy(), geometry=geometry, crs="EPSG:4326")
    if target_crs is not None and stations_gdf.crs != target_crs:
        stations_gdf = stations_gdf.to_crs(target_crs)
    return stations_gdf


def main():
    print("Loading census tracts GeoJSON...")
    tracts_gdf = load_census_tracts(TRACTS_PATH)

    print("Loading stations CSV...")
    stations_df = load_stations(STATIONS_PATH)

    print("Preparing station geometries and aligning CRS...")
    stations_gdf = convert_stations_to_gdf(stations_df, tracts_gdf.crs)

    # Select only key census columns that actually exist in the tracts GeoJSON
    available_id_cols = [c for c in CENSUS_ID_COLS if c in tracts_gdf.columns]
    available_value_cols = [c for c in CENSUS_VALUE_COLS if c in tracts_gdf.columns]
    census_cols = available_id_cols + available_value_cols

    print("Performing spatial join (stations within tracts)...")
    joined = gpd.sjoin(stations_gdf, tracts_gdf[census_cols + ["geometry"]], how="left", predicate="within")

    # Clean up: drop spatial join helper columns
    for col in ["index_right"]:
        if col in joined.columns:
            joined.drop(columns=[col], inplace=True)

    # Prepare output: only station_id, station_name + census cols
    base_cols = [c for c in ["station_id", "station_name"] if c in stations_df.columns]
    output_cols = base_cols + census_cols
    output_df = joined[output_cols].copy()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Writing output CSV to: {OUTPUT_PATH}")
    output_df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
