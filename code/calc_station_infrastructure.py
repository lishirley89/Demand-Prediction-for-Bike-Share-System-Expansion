#!/usr/bin/env python3
"""
Calculate infrastructure metrics within each station's 500m buffer.

Loads:
- Station buffers: result/master_stations_buffer_500m.geojson
- Bike routes: data/Bike_Routes_20250828.geojson  
- Rail stations: data/CTA_-_'L'_(Rail)_Stations_20250828.geojson
- Bus stops: data/CTA_BusStops_20250828.geojson
- Street centerlines: data/transportation_streetcenterlines20250828.geojson

Outputs:
- result/station_infrastructure_metrics.csv with columns:
  - station_id: Station identifier
  - station_name: Station name
  - bike_route_length_m: Total length of bike routes within buffer (meters)
  - street_length_m: Total length of street centerlines within buffer (meters)  
  - rail_stops_count: Number of rail station stops within buffer
  - bus_stops_count: Number of bus stops within buffer
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


# Project paths
PROJECT_ROOT = Path('/Users/shirley/Documents/Research/demand/parkchargebike')
RESULT_DIR = PROJECT_ROOT / 'result'
DATA_DIR = PROJECT_ROOT / 'data'

# Input files
STATION_BUFFERS_GEOJSON = RESULT_DIR / 'master_stations_buffer_500m.geojson'
BIKE_ROUTES_GEOJSON = DATA_DIR / 'Bike_Routes_20250828.geojson'
RAIL_STATIONS_GEOJSON = DATA_DIR / "CTA_-_'L'_(Rail)_Stations_20250828.geojson"
BUS_STOPS_GEOJSON = DATA_DIR / 'CTA_BusStops_20250828.geojson'
STREET_CENTERLINES_GEOJSON = DATA_DIR / 'transportation_streetcenterlines20250828.geojson'

# Output file
OUTPUT_CSV = RESULT_DIR / 'station_infrastructure_metrics.csv'


def load_geodata() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load all geospatial data files."""
    print("Loading station buffers...")
    station_buffers = gpd.read_file(STATION_BUFFERS_GEOJSON)
    
    print("Loading bike routes...")
    bike_routes = gpd.read_file(BIKE_ROUTES_GEOJSON)
    
    print("Loading rail stations...")
    rail_stations = gpd.read_file(RAIL_STATIONS_GEOJSON)
    
    print("Loading bus stops...")
    bus_stops = gpd.read_file(BUS_STOPS_GEOJSON)
    
    print("Loading street centerlines...")
    street_centerlines = gpd.read_file(STREET_CENTERLINES_GEOJSON)
    
    return station_buffers, bike_routes, rail_stations, bus_stops, street_centerlines


def ensure_consistent_crs(gdfs: list[gpd.GeoDataFrame]) -> list[gpd.GeoDataFrame]:
    """Ensure all GeoDataFrames have the same CRS (use the first one's CRS)."""
    target_crs = gdfs[0].crs
    print(f"Using CRS: {target_crs}")
    
    result = []
    for i, gdf in enumerate(gdfs):
        if gdf.crs != target_crs:
            print(f"Reprojecting dataset {i+1} from {gdf.crs} to {target_crs}")
            gdf = gdf.to_crs(target_crs)
        result.append(gdf)
    
    return result


def calculate_bike_route_length(station_buffers: gpd.GeoDataFrame, bike_routes: gpd.GeoDataFrame) -> pd.Series:
    """Calculate total bike route length within each station buffer."""
    print("Calculating bike route lengths...")
    
    # Ensure we're working with projected coordinates for accurate length calculation
    if station_buffers.crs.is_geographic:
        print("Reprojecting to UTM 16N for accurate length calculations...")
        station_buffers_proj = station_buffers.to_crs(epsg=32616)
        bike_routes_proj = bike_routes.to_crs(epsg=32616)
    else:
        station_buffers_proj = station_buffers
        bike_routes_proj = bike_routes
    
    bike_lengths = []
    
    for idx, buffer_geom in station_buffers_proj.geometry.items():
        # Find bike routes that intersect with this buffer
        intersecting_routes = bike_routes_proj[bike_routes_proj.geometry.intersects(buffer_geom)]
        
        if len(intersecting_routes) == 0:
            bike_lengths.append(0.0)
        else:
            # Calculate intersection length for each route
            total_length = 0.0
            for _, route in intersecting_routes.iterrows():
                intersection = route.geometry.intersection(buffer_geom)
                if not intersection.is_empty:
                    total_length += intersection.length
            bike_lengths.append(total_length)
    
    return pd.Series(bike_lengths, index=station_buffers.index)


def calculate_street_length(station_buffers: gpd.GeoDataFrame, street_centerlines: gpd.GeoDataFrame) -> pd.Series:
    """Calculate total street centerline length within each station buffer."""
    print("Calculating street centerline lengths...")
    
    # Ensure we're working with projected coordinates for accurate length calculation
    if station_buffers.crs.is_geographic:
        print("Reprojecting to UTM 16N for accurate length calculations...")
        station_buffers_proj = station_buffers.to_crs(epsg=32616)
        street_centerlines_proj = street_centerlines.to_crs(epsg=32616)
    else:
        station_buffers_proj = station_buffers
        street_centerlines_proj = street_centerlines
    
    street_lengths = []
    
    for idx, buffer_geom in station_buffers_proj.geometry.items():
        # Find street centerlines that intersect with this buffer
        intersecting_streets = street_centerlines_proj[street_centerlines_proj.geometry.intersects(buffer_geom)]
        
        if len(intersecting_streets) == 0:
            street_lengths.append(0.0)
        else:
            # Calculate intersection length for each street
            total_length = 0.0
            for _, street in intersecting_streets.iterrows():
                intersection = street.geometry.intersection(buffer_geom)
                if not intersection.is_empty:
                    total_length += intersection.length
            street_lengths.append(total_length)
    
    return pd.Series(street_lengths, index=station_buffers.index)


def count_points_in_buffers(station_buffers: gpd.GeoDataFrame, points_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Count number of points within each station buffer."""
    counts = []
    
    for idx, buffer_geom in station_buffers.geometry.items():
        # Count points that are within this buffer
        points_within = points_gdf[points_gdf.geometry.within(buffer_geom)]
        counts.append(len(points_within))
    
    return pd.Series(counts, index=station_buffers.index)


def main():
    """Main function to calculate and save infrastructure metrics."""
    print("Starting infrastructure metrics calculation...")
    
    # Load all geospatial data
    station_buffers, bike_routes, rail_stations, bus_stops, street_centerlines = load_geodata()
    
    # Ensure consistent CRS
    station_buffers, bike_routes, rail_stations, bus_stops, street_centerlines = ensure_consistent_crs([
        station_buffers, bike_routes, rail_stations, bus_stops, street_centerlines
    ])
    
    print(f"Processing {len(station_buffers)} station buffers...")
    
    # Calculate bike route lengths
    bike_lengths = calculate_bike_route_length(station_buffers, bike_routes)
    
    # Calculate street centerline lengths  
    street_lengths = calculate_street_length(station_buffers, street_centerlines)
    
    # Count rail stations
    print("Counting rail stations...")
    rail_counts = count_points_in_buffers(station_buffers, rail_stations)
    
    # Count bus stops
    print("Counting bus stops...")
    bus_counts = count_points_in_buffers(station_buffers, bus_stops)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'station_id': station_buffers['station_id'],
        'station_name': station_buffers['station_name'],
        'bike_route_length_m': bike_lengths,
        'street_length_m': street_lengths,
        'rail_stops_count': rail_counts,
        'bus_stops_count': bus_counts
    })
    
    # Ensure output directory exists
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total stations processed: {len(results)}")
    print(f"Average bike route length: {results['bike_route_length_m'].mean():.1f} meters")
    print(f"Average street length: {results['street_length_m'].mean():.1f} meters")
    print(f"Average rail stops per buffer: {results['rail_stops_count'].mean():.1f}")
    print(f"Average bus stops per buffer: {results['bus_stops_count'].mean():.1f}")


if __name__ == '__main__':
    main()
