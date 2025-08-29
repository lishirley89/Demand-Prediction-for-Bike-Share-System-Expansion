#!/usr/bin/env python3
"""
Create 500 meter buffers for each station in result/master_stations.csv
and write them to result/master_stations_buffer_500m.geojson.

This script does NOT modify result/master_stations.csv.

Approach:
- Read stations as WGS84 (EPSG:4326)
- Project to a local metric CRS (UTM zone 16N, EPSG:32616 for Chicago area)
- Buffer by 500 meters
- Reproject buffers back to EPSG:4326 and write GeoJSON
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


PROJECT_ROOT = Path('/Users/shirley/Documents/Research/demand/parkchargebike')
RESULT_DIR = PROJECT_ROOT / 'result'
MASTER_STATIONS_CSV = RESULT_DIR / 'master_stations.csv'
OUTPUT_GEOJSON = RESULT_DIR / 'master_stations_buffer_500m.geojson'


def load_stations(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path, dtype={'station_id': str})
    # Ensure numeric and drop invalid rows
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng'])

    # Create point geometries in WGS84
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df['lng'], df['lat'])],
        crs='EPSG:4326',
    )
    return gdf


def create_500m_buffers(stations_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Use UTM 16N (meters) for Chicago region buffering
    stations_proj = stations_wgs84.to_crs(epsg=32616)
    stations_proj['geometry'] = stations_proj.geometry.buffer(500.0)
    buffers_wgs84 = stations_proj.to_crs(epsg=4326)
    # Keep essential columns
    keep_cols = ['station_id', 'station_name']
    keep_cols = [c for c in keep_cols if c in buffers_wgs84.columns]
    buffers_wgs84 = buffers_wgs84[keep_cols + ['geometry']]
    return buffers_wgs84


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_stations(MASTER_STATIONS_CSV)
    buffers = create_500m_buffers(stations)
    buffers.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
    print(f"Wrote {len(buffers)} 500m buffers to {OUTPUT_GEOJSON}")


if __name__ == '__main__':
    main()


