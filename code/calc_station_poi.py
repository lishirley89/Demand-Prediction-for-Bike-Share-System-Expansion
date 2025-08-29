#!/usr/bin/env python3
"""
Calculate the number of Points of Interest (POI) of different categories 
within each station's 500m buffer using multiple ArcGIS REST services.

Uses the following services:
- Tourism: https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Tourism/FeatureServer/0
- Educational: https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Educational/FeatureServer/0
- Medical: https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Medical/FeatureServer/0
- Shops: https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Shops/FeatureServer/0
- Leisure: https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Leisure/FeatureServer/0

Outputs:
- result/station_poi_counts.csv with columns:
  - station_id: Station identifier
  - station_name: Station name
  - poi_tourism: Total number of tourism POIs within buffer
  - poi_education: Total number of educational POIs within buffer
  - poi_medical: Total number of medical POIs within buffer
  - poi_shop: Total number of shop POIs within buffer
  - poi_leisure: Total number of leisure POIs within buffer
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import pyproj
import requests
from shapely.geometry import Point, box
from shapely.ops import transform


# Project paths
PROJECT_ROOT = Path('/Users/shirley/Documents/Research/demand/parkchargebike')
RESULT_DIR = PROJECT_ROOT / 'result'
DATA_DIR = PROJECT_ROOT / 'data'

# Input files
STATION_BUFFERS_GEOJSON = RESULT_DIR / 'master_stations_buffer_500m.geojson'

# Output file
OUTPUT_CSV = RESULT_DIR / 'master_station_poi_counts.csv'

# ArcGIS REST service URLs
ARCGIS_SERVICES = {
    'tourism': "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Tourism/FeatureServer/0",
    'education': "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Educational/FeatureServer/0",
    'medical': "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Medical/FeatureServer/0",
    'shop': "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Shops/FeatureServer/0",
    'leisure': "https://services6.arcgis.com/Do88DoK2xjTUCXd1/ArcGIS/rest/services/OSM_NA_Leisure/FeatureServer/0"
}


def load_station_buffers() -> gpd.GeoDataFrame:
    """Load station buffers from GeoJSON."""
    print("Loading station buffers...")
    buffers = gpd.read_file(STATION_BUFFERS_GEOJSON)
    return buffers


def get_poi_data_for_bbox(service_url: str, bbox: tuple, max_records: int = 2000) -> List[Dict]:
    """
    Query ArcGIS REST service for POIs within a bounding box.
    
    Args:
        service_url: ArcGIS REST service URL
        bbox: (xmin, ymin, xmax, ymax) in WGS84 coordinates
        max_records: Maximum number of records to return
    
    Returns:
        List of POI features
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Create coordinate transformer from WGS84 to Web Mercator
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    
    # Transform bounding box coordinates
    xmin_merc, ymin_merc = transformer.transform(xmin, ymin)
    xmax_merc, ymax_merc = transformer.transform(xmax, ymax)
    
    # ArcGIS REST query parameters
    params = {
        'f': 'json',
        'where': '1=1',  # Get all features
        'geometry': json.dumps({
            'xmin': xmin_merc,
            'ymin': ymin_merc,
            'xmax': xmax_merc,
            'ymax': ymax_merc,
            'spatialReference': {'wkid': 3857}
        }),
        'geometryType': 'esriGeometryEnvelope',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': '*',  # Get all fields
        'returnGeometry': 'true',
        'maxRecordCount': max_records
    }
    
    try:
        response = requests.get(service_url + '/query', params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'features' in data:
            print(f"    Found {len(data['features'])} POIs in bbox")
            return data['features']
        else:
            print(f"    Warning: No features found in response for bbox {bbox}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"    Error querying ArcGIS service for bbox {bbox}: {e}")
        return []


def count_pois_in_buffer_by_service(buffer_geom, buffer_bbox: tuple, service_url: str) -> int:
    """
    Count total POIs from a specific service within a station buffer.
    
    Args:
        buffer_geom: Shapely geometry of the buffer
        buffer_bbox: Bounding box of the buffer (xmin, ymin, xmax, ymax)
        service_url: ArcGIS REST service URL
    
    Returns:
        Total count of POIs from this service within the buffer
    """
    # Get POI data for the bounding box
    poi_features = get_poi_data_for_bbox(service_url, buffer_bbox)
    
    count = 0
    
    # Create coordinate transformer from Web Mercator to WGS84
    transformer_merc_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    
    # Process each POI feature
    for feature in poi_features:
        if 'geometry' in feature:
            # Create POI point geometry
            geom = feature['geometry']
            if 'x' in geom and 'y' in geom:
                # Transform Web Mercator coordinates back to WGS84 for comparison
                x_wgs84, y_wgs84 = transformer_merc_to_wgs84.transform(geom['x'], geom['y'])
                poi_point = Point(x_wgs84, y_wgs84)
                
                # Check if POI is within the buffer
                if buffer_geom.contains(poi_point):
                    count += 1
    
    return count


def count_all_pois_in_buffer(buffer_geom, buffer_bbox: tuple) -> Dict[str, int]:
    """
    Count POIs from all services within a station buffer.
    
    Args:
        buffer_geom: Shapely geometry of the buffer
        buffer_bbox: Bounding box of the buffer (xmin, ymin, xmax, ymax)
    
    Returns:
        Dictionary with POI counts by service category
    """
    counts = {}
    
    # Count POIs from each service
    for service_name, service_url in ARCGIS_SERVICES.items():
        print(f"  Querying {service_name} POIs...")
        count = count_pois_in_buffer_by_service(buffer_geom, buffer_bbox, service_url)
        counts[f'poi_{service_name}'] = count
    
    return counts


def main():
    """Main function to calculate POI counts for all station buffers."""
    print("Starting POI count calculation...")
    
    # Load station buffers
    station_buffers = load_station_buffers()
    
    print(f"Processing {len(station_buffers)} station buffers...")
    
    # Initialize results list
    results = []
    
    # Process each station buffer
    for idx, (_, row) in enumerate(station_buffers.iterrows()):
        station_id = row['station_id']
        station_name = row['station_name']
        buffer_geom = row.geometry
        
        print(f"Processing station {idx+1}/{len(station_buffers)}: {station_name}")
        
        # Get buffer bounding box
        bbox = buffer_geom.bounds  # (xmin, ymin, xmax, ymax)
        
        # Count POIs in buffer from all services
        poi_counts = count_all_pois_in_buffer(buffer_geom, bbox)
        
        # Add station info to results
        result_row = {
            'station_id': station_id,
            'station_name': station_name,
            **poi_counts
        }
        results.append(result_row)
        
        # Add small delay to avoid overwhelming the service
        time.sleep(0.1)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total stations processed: {len(results_df)}")
    print(f"Average tourism POIs per buffer: {results_df['poi_tourism'].mean():.1f}")
    print(f"Average education POIs per buffer: {results_df['poi_education'].mean():.1f}")
    print(f"Average medical POIs per buffer: {results_df['poi_medical'].mean():.1f}")
    print(f"Average shop POIs per buffer: {results_df['poi_shop'].mean():.1f}")
    print(f"Average leisure POIs per buffer: {results_df['poi_leisure'].mean():.1f}")


if __name__ == '__main__':
    main()
