#!/usr/bin/env python3
"""
Generate H3 cells at resolution 8 for Chicago and create CSV with cell indices and centroids.

Output: result/chicago_h3_res8.csv with columns: h3_index, lat, lng
"""
import h3
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Configuration
RESOLUTION = 8
CHICAGO_BOUNDARY = Path("data/Boundaries_-_City_20251224.geojson")
OUTPUT_CSV = Path("result/chicago_h3_res8.csv")


def polyfill_shapely(geom, res):
    """Convert Shapely geometry to H3 cells using h3-py 4.x API."""
    hexes = set()

    if isinstance(geom, (gpd.GeoSeries, gpd.GeoDataFrame)):
        geom = geom.geometry.iloc[0] if hasattr(geom, 'geometry') else geom

    from shapely.geometry import Polygon, MultiPolygon

    if isinstance(geom, Polygon):
        geo_dict = geom.__geo_interface__
        hexes.update(h3.geo_to_cells(geo_dict, res))

    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            geo_dict = poly.__geo_interface__
            hexes.update(h3.geo_to_cells(geo_dict, res))

    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

    return hexes


def main():
    """Generate H3 cells and save to CSV."""
    print(f"Loading Chicago boundary from {CHICAGO_BOUNDARY}...")
    chicago = gpd.read_file(CHICAGO_BOUNDARY)
    geom = chicago.geometry.iloc[0]

    print(f"Generating H3 cells at resolution {RESOLUTION}...")
    hexes = polyfill_shapely(geom, RESOLUTION)
    print(f"Found {len(hexes)} H3 cells")

    print("Calculating centroids...")
    results = []
    for h3_index in sorted(hexes):
        lat, lng = h3.cell_to_latlng(h3_index)
        results.append({
            "h3_index": h3_index,
            "lat": lat,
            "lng": lng
        })

    df = pd.DataFrame(results)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} H3 cells to {OUTPUT_CSV}")

    print(f"\nSample of first 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()