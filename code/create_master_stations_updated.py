#!/usr/bin/env python3
"""
Create a master station file from all trip data files.
Analyzes station IDs and names to identify any duplicates or inconsistencies.
Uses the first ID for stations with multiple IDs.
Creates additional files for daily trip counts by type.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def create_master_stations():
    """Create master station file from all trip data."""
    
    # Setup paths
    PROJECT_ROOT = Path('/Users/shirley/Documents/Research/demand/parkchargebike')
    TRIP_DIR = PROJECT_ROOT / 'trip'
    RESULT_DIR = PROJECT_ROOT / 'result'
    
    # Create result directory if it doesn't exist
    RESULT_DIR.mkdir(exist_ok=True)
    
    # Find all station summary files
    station_files = list(TRIP_DIR.glob('*-station-summary.csv'))
    print(f"Found {len(station_files)} station summary files")
    
    # Collect all unique stations
    all_stations = {}
    station_name_to_ids = defaultdict(set)
    station_id_to_names = {}
    station_name_to_first_id = {}  # Track first ID for each station name
    
    for file_path in station_files:
        print(f"Processing {file_path.name}...")
        
        # Read the file
        df = pd.read_csv(file_path)
        
        # Extract year and month from filename
        filename = file_path.stem  # Remove .csv extension
        year_month = filename.split('-')[0]
        year = int(year_month[:4])
        month = int(year_month[4:6])
        
        # Process each station
        for _, row in df.iterrows():
            station_id = row['station_id']
            station_name = row['station_name']
            lat = row['lat']
            lng = row['lng']
            
            # Track station name to IDs mapping
            station_name_to_ids[station_name].add(station_id)
            
            # Track first ID for each station name
            if station_name not in station_name_to_first_id:
                station_name_to_first_id[station_name] = station_id
            
            # Track station ID to name mapping
            if station_id not in station_id_to_names:
                station_id_to_names[station_id] = station_name
            elif station_id_to_names[station_id] != station_name:
                print(f"WARNING: Station ID {station_id} has different names: '{station_id_to_names[station_id]}' vs '{station_name}'")
            
            # Use first ID for stations with multiple IDs
            primary_station_id = station_name_to_first_id[station_name]
            
            # Store station info using primary ID
            if primary_station_id not in all_stations:
                all_stations[primary_station_id] = {
                    'station_id': primary_station_id,
                    'station_name': station_name,
                    'lat': lat,
                    'lng': lng,
                    'first_seen': f"{year}-{month:02d}",
                    'last_seen': f"{year}-{month:02d}",
                    'total_months': 1,
                    'monthly_activity': {},  # Track monthly activity
                    'monthly_ebike_start': {},  # Track daily ebike starts
                    'monthly_cbike_start': {},  # Track daily cbike starts
                    'monthly_total_start': {},  # Track daily total starts
                    'monthly_ebike_end': {},  # Track daily ebike ends
                    'monthly_cbike_end': {},  # Track daily cbike ends
                    'monthly_total_end': {}  # Track daily total ends
                }
            
            # Update monthly activity
            month_key = f"{year:04d}{month:02d}"
            all_stations[primary_station_id]['monthly_activity'][month_key] = 1
            
            # Update daily trip counts for this month
            if month_key not in all_stations[primary_station_id]['monthly_ebike_start']:
                all_stations[primary_station_id]['monthly_ebike_start'][month_key] = row['daily_ebike_start']
                all_stations[primary_station_id]['monthly_cbike_start'][month_key] = row['daily_cbike_start']
                all_stations[primary_station_id]['monthly_total_start'][month_key] = row['daily_start']
                all_stations[primary_station_id]['monthly_ebike_end'][month_key] = row['daily_ebike_end']
                all_stations[primary_station_id]['monthly_cbike_end'][month_key] = row['daily_cbike_end']
                all_stations[primary_station_id]['monthly_total_end'][month_key] = row['daily_end']
            
            # Update last seen date
            current_last = all_stations[primary_station_id]['last_seen']
            new_date = f"{year}-{month:02d}"
            if new_date > current_last:
                all_stations[primary_station_id]['last_seen'] = new_date
            
            # Update first seen date if earlier
            current_first = all_stations[primary_station_id]['first_seen']
            if new_date < current_first:
                all_stations[primary_station_id]['first_seen'] = new_date
    
    # Get all unique month keys for column creation
    all_month_keys = set()
    for station in all_stations.values():
        all_month_keys.update(station['monthly_activity'].keys())
    
    # Sort month keys chronologically
    sorted_month_keys = sorted(all_month_keys)
    
    # Convert to DataFrame for master stations
    master_stations_list = []
    for station in all_stations.values():
        station_data = {
            'station_id': station['station_id'],
            'station_name': station['station_name'],
            'lat': station['lat'],
            'lng': station['lng'],
            'first_seen': station['first_seen'],
            'last_seen': station['last_seen'],
            'total_months': len(station['monthly_activity'])
        }
        
        # Add monthly activity columns
        for month_key in sorted_month_keys:
            station_data[month_key] = station['monthly_activity'].get(month_key, 0)
        
        master_stations_list.append(station_data)
    
    master_stations = pd.DataFrame(master_stations_list)
    master_stations = master_stations.sort_values('station_id')
    
    # Create daily trip count DataFrames
    daily_ebike_start_list = []
    daily_cbike_start_list = []
    daily_total_start_list = []
    daily_ebike_end_list = []
    daily_cbike_end_list = []
    daily_total_end_list = []
    
    for station in all_stations.values():
        # Base station info
        base_data = {
            'station_id': station['station_id'],
            'station_name': station['station_name'],
            'lat': station['lat'],
            'lng': station['lng']
        }
        
        # Daily ebike start
        ebike_start_data = base_data.copy()
        for month_key in sorted_month_keys:
            ebike_start_data[month_key] = station['monthly_ebike_start'].get(month_key, 0.0)
        daily_ebike_start_list.append(ebike_start_data)
        
        # Daily cbike start
        cbike_start_data = base_data.copy()
        for month_key in sorted_month_keys:
            cbike_start_data[month_key] = station['monthly_cbike_start'].get(month_key, 0.0)
        daily_cbike_start_list.append(cbike_start_data)
        
        # Daily total start
        total_start_data = base_data.copy()
        for month_key in sorted_month_keys:
            total_start_data[month_key] = station['monthly_total_start'].get(month_key, 0.0)
        daily_total_start_list.append(total_start_data)
        
        # Daily ebike end
        ebike_end_data = base_data.copy()
        for month_key in sorted_month_keys:
            ebike_end_data[month_key] = station['monthly_ebike_end'].get(month_key, 0.0)
        daily_ebike_end_list.append(ebike_end_data)
        
        # Daily cbike end
        cbike_end_data = base_data.copy()
        for month_key in sorted_month_keys:
            cbike_end_data[month_key] = station['monthly_cbike_end'].get(month_key, 0.0)
        daily_cbike_end_list.append(cbike_end_data)
        
        # Daily total end
        total_end_data = base_data.copy()
        for month_key in sorted_month_keys:
            total_end_data[month_key] = station['monthly_total_end'].get(month_key, 0.0)
        daily_total_end_list.append(total_end_data)
    
    # Create DataFrames
    daily_ebike_start_df = pd.DataFrame(daily_ebike_start_list).sort_values('station_id')
    daily_cbike_start_df = pd.DataFrame(daily_cbike_start_list).sort_values('station_id')
    daily_total_start_df = pd.DataFrame(daily_total_start_list).sort_values('station_id')
    daily_ebike_end_df = pd.DataFrame(daily_ebike_end_list).sort_values('station_id')
    daily_cbike_end_df = pd.DataFrame(daily_cbike_end_list).sort_values('station_id')
    daily_total_end_df = pd.DataFrame(daily_total_end_list).sort_values('station_id')
    
    # Save all files
    master_stations.to_csv(RESULT_DIR / 'master_stations.csv', index=False)
    daily_ebike_start_df.to_csv(RESULT_DIR / 'master_stations_daily_ebike_start.csv', index=False)
    daily_cbike_start_df.to_csv(RESULT_DIR / 'master_stations_daily_cbike_start.csv', index=False)
    daily_total_start_df.to_csv(RESULT_DIR / 'master_stations_daily_start.csv', index=False)
    daily_ebike_end_df.to_csv(RESULT_DIR / 'master_stations_daily_ebike_end.csv', index=False)
    daily_cbike_end_df.to_csv(RESULT_DIR / 'master_stations_daily_cbike_end.csv', index=False)
    daily_total_end_df.to_csv(RESULT_DIR / 'master_stations_daily_end.csv', index=False)
    
    print(f"\nFiles saved to {RESULT_DIR}:")
    print(f"  - master_stations.csv")
    print(f"  - master_stations_daily_ebike_start.csv")
    print(f"  - master_stations_daily_cbike_start.csv")
    print(f"  - master_stations_daily_start.csv")
    print(f"  - master_stations_daily_ebike_end.csv")
    print(f"  - master_stations_daily_cbike_end.csv")
    print(f"  - master_stations_daily_end.csv")
    
    print(f"\nTotal unique stations: {len(master_stations)}")
    
    # Analyze station name to ID mappings
    print("\n" + "="*60)
    print("STATION NAME TO ID ANALYSIS")
    print("="*60)
    
    # Find stations with multiple IDs
    multiple_ids = {name: ids for name, ids in station_name_to_ids.items() if len(ids) > 1}
    
    if multiple_ids:
        print(f"Found {len(multiple_ids)} station names with multiple IDs:")
        for name, ids in sorted(multiple_ids.items()):
            print(f"\nStation Name: '{name}'")
            print(f"  IDs: {sorted(ids)}")
            first_id = station_name_to_first_id[name]
            print(f"  Using first ID: {first_id}")
    else:
        print("No station names found with multiple IDs.")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total unique station names: {len(station_name_to_ids)}")
    print(f"Total unique station IDs (using first ID for duplicates): {len(master_stations)}")
    print(f"Stations with multiple IDs: {len(multiple_ids)}")
    print(f"Monthly activity columns: {len(sorted_month_keys)}")
    print(f"Date range: {sorted_month_keys[0]} to {sorted_month_keys[-1]}")
    
    # Show sample of master stations
    print("\n" + "="*60)
    print("SAMPLE OF MASTER STATIONS")
    print("="*60)
    print(master_stations.head(5))
    
    # Save analysis results
    if multiple_ids:
        multiple_ids_df = pd.DataFrame([
            {'station_name': name, 'station_ids': ', '.join(sorted(ids)), 'first_id': station_name_to_first_id[name]}
            for name, ids in multiple_ids.items()
        ])
        multiple_ids_path = RESULT_DIR / 'stations_with_multiple_ids.csv'
        multiple_ids_df.to_csv(multiple_ids_path, index=False)
        print(f"\nStations with multiple IDs saved to: {multiple_ids_path}")
    
    # Save summary statistics
    summary_stats = {
        'metric': ['Total unique station names', 'Total unique station IDs (using first ID)', 
                  'Stations with multiple IDs', 'Monthly activity columns'],
        'count': [len(station_name_to_ids), len(master_stations), 
                 len(multiple_ids), len(sorted_month_keys)]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_path = RESULT_DIR / 'station_analysis_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    return master_stations, multiple_ids

if __name__ == "__main__":
    master_stations, multiple_ids = create_master_stations()
