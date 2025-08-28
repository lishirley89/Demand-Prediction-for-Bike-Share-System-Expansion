#!/usr/bin/env python3
"""
Compute the full great-circle distance matrix (km) between all stations
listed in result/master_stations.csv using the haversine formula.

Output CSV: result/master_stations_dist_km.csv
 - Row index: station_id (also a column in the file header as index column)
 - Columns: station_id for each station
 - Values: distance in kilometers

Usage:
  python code/calc_dist.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path('/Users/shirley/Documents/Research/demand/parkchargebike')
RESULT_DIR = PROJECT_ROOT / 'result'
MASTER_STATIONS_CSV = RESULT_DIR / 'master_stations.csv'
OUTPUT_CSV = RESULT_DIR / 'master_stations_dist_km.csv'


def haversine_matrix_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vectorized haversine to compute an NxN distance matrix (km).

    Parameters
    - lat: array of shape (N,)
    - lon: array of shape (N,)
    Returns
    - distances: array of shape (N, N)
    """
    lat_rad = np.radians(lat).reshape(-1, 1)
    lon_rad = np.radians(lon).reshape(-1, 1)

    dlat = lat_rad - lat_rad.T
    dlon = lon_rad - lon_rad.T

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(lat_rad.T) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0088  # IUGG mean Earth radius in kilometers
    return earth_radius_km * c


def load_stations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={'station_id': str})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng'])
    # Keep only needed columns and stable order
    df = df[['station_id', 'lat', 'lng']].copy()
    return df


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    stations = load_stations(MASTER_STATIONS_CSV)
    station_ids = stations['station_id'].tolist()
    lat = stations['lat'].to_numpy()
    lon = stations['lng'].to_numpy()

    dist_matrix = haversine_matrix_km(lat, lon)

    # Build DataFrame with station_id as both index and columns
    mat_df = pd.DataFrame(dist_matrix, index=station_ids, columns=station_ids)
    mat_df.insert(0, 'station_id', mat_df.index)

    # Write CSV with index=True so row number is present, station_id is first column
    mat_df.to_csv(OUTPUT_CSV, index=True)


if __name__ == '__main__':
    main()


