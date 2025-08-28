#!/usr/bin/env python3
"""
Calculate the number of active stations within different distance ranges for each month.

For each month, get active stations (value 1 in master_stations.csv) and count how many
are within 0-500m, 500-1000m, and 1000-1500m using distances from master_stations_dist_km.csv.

Output files:
- result/master_stations_500m.csv: stations within 0-500m
- result/master_stations_1000m.csv: stations within 500-1000m  
- result/master_stations_1500m.csv: stations within 1000-1500m

Each CSV has station_id as row index and months as columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path('/Users/shirley/Documents/Research/demand/parkchargebike')
RESULT_DIR = PROJECT_ROOT / 'result'
MASTER_STATIONS_CSV = RESULT_DIR / 'master_stations.csv'
DISTANCE_MATRIX_CSV = RESULT_DIR / 'master_stations_dist_km.csv'

# Distance ranges in kilometers
DISTANCE_RANGES = {
    '500m': (0, 0.5),
    '1000m': (0.5, 1.0), 
    '1500m': (1.0, 1.5)
}

def load_data():
    """Load master stations and distance matrix data."""
    print("Loading master stations data...")
    stations_df = pd.read_csv(MASTER_STATIONS_CSV, dtype={'station_id': str})
    
    print("Loading distance matrix...")
    # Read distance matrix with station_id as index
    dist_df = pd.read_csv(DISTANCE_MATRIX_CSV, index_col=0)
    
    # Convert all columns to numeric (they should all be distances)
    dist_df = dist_df.apply(pd.to_numeric, errors='coerce')
    
    return stations_df, dist_df

def get_month_columns(stations_df):
    """Get list of month columns (YYYYMM format)."""
    month_cols = [col for col in stations_df.columns if col.isdigit() and len(col) == 6]
    return sorted(month_cols)

def calculate_active_stations_by_distance(stations_df, dist_df, distance_range, month_cols):
    """Calculate active stations within distance range for each month."""
    min_dist, max_dist = distance_range
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=dist_df.index, columns=month_cols)
    results.index.name = 'station_id'
    
    print(f"Processing distance range {min_dist}-{max_dist} km for {len(month_cols)} months...")
    
    for month in month_cols:
        print(f"  Processing {month}...")
        
        # Get active stations for this month
        active_stations = stations_df[stations_df[month] == 1]['station_id'].astype(str).tolist()
        
        # For each station, count active stations within distance range
        for station_id in dist_df.index:
            # Check if this station is active in this month
            station_active = station_id in active_stations
            
            if not station_active:
                # If station is not active, set to null
                results.loc[station_id, month] = np.nan
            else:
                # Get distances from this station to all other stations
                distances = dist_df.loc[station_id]
                
                # Find stations within distance range that are also active
                within_range = (distances >= min_dist) & (distances <= max_dist)
                active_within_range = [s for s in within_range[within_range].index if s in active_stations]
                
                # Count (excluding self)
                count = len([s for s in active_within_range if s != station_id])
                results.loc[station_id, month] = count
    
    return results

def main():
    """Main function to calculate and save results."""
    # Load data
    stations_df, dist_df = load_data()
    
    # Get month columns
    month_cols = get_month_columns(stations_df)
    print(f"Found {len(month_cols)} months: {month_cols[:5]}...{month_cols[-5:]}")
    
    # Ensure output directory exists
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate for each distance range
    for range_name, distance_range in DISTANCE_RANGES.items():
        print(f"\nCalculating {range_name} range...")
        
        results = calculate_active_stations_by_distance(
            stations_df, dist_df, distance_range, month_cols
        )
        
        # Save results
        output_file = RESULT_DIR / f'master_stations_{range_name}.csv'
        results.to_csv(output_file)
        print(f"Saved {output_file}")
        
        # Print some statistics
        print(f"  Average active stations per month: {results.mean().mean():.1f}")
        print(f"  Max active stations in any month: {results.max().max()}")
        print(f"  Min active stations in any month: {results.min().min()}")

if __name__ == '__main__':
    main()
