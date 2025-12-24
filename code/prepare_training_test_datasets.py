#!/usr/bin/env python3
"""
Script to prepare training and test datasets for bike station analysis.

This script creates individual rows for each active station in each month:
- One row per active station per month
- All individual values (not averages) for infrastructure, census, POI, and trip data
- Training dataset: 202101 to 202407
- Test dataset: 202408 to 202507

Each row contains:
- month, year (e.g., 01, 2021)
- n250, n500, n750, n1000, n1250, n1500 (from distance range files)
- All infrastructure metrics (bike_route_length_m, street_length_m, etc.)
- All POI counts (poi_tourism, poi_education, etc.)
- All census variables (pct_white, pct_black, etc.)
- All trip data (cbike_start, cbike_end, etc.)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_data_files():
    """Load all required data files."""
    print("Loading data files...")
    
    # Load training stations (active stations by month)
    training_stations = pd.read_csv("result/master_stations_fortraining.csv")
    print(f"Loaded {len(training_stations)} training stations")
    
    # Load distance range files
    distance_files = {
        'n250': "result/master_stations_0_250m.csv",
        'n500': "result/master_stations_250_500m.csv", 
        'n750': "result/master_stations_500_750m.csv",
        'n1000': "result/master_stations_750_1000m.csv",
        'n1250': "result/master_stations_1000_1250m.csv",
        'n1500': "result/master_stations_1250_1500m.csv"
    }
    
    distance_data = {}
    for key, filepath in distance_files.items():
        if os.path.exists(filepath):
            distance_data[key] = pd.read_csv(filepath)
            print(f"Loaded {key}: {filepath}")
        else:
            print(f"Warning: {filepath} not found")
    
    # Load POI counts
    poi_data = pd.read_csv("result/master_station_poi_counts.csv")
    print(f"Loaded POI data: {len(poi_data)} stations")
    
    # Load infrastructure metrics
    infra_data = pd.read_csv("result/master_station_infrastructure_metrics.csv")
    print(f"Loaded infrastructure data: {len(infra_data)} stations")
    
    # Load census data
    census_data = pd.read_csv("result/master_stations_census.csv")
    print(f"Loaded census data: {len(census_data)} stations")
    
    # Load daily trip data files
    daily_files = {
        'cbike_start': "result/master_stations_daily_cbike_start.csv",
        'cbike_end': "result/master_stations_daily_cbike_end.csv",
        'ebike_start': "result/master_stations_daily_ebike_start.csv",
        'ebike_end': "result/master_stations_daily_ebike_end.csv",
        'total_start': "result/master_stations_daily_start.csv",
        'total_end': "result/master_stations_daily_end.csv"
    }
    
    daily_data = {}
    for key, filepath in daily_files.items():
        if os.path.exists(filepath):
            daily_data[key] = pd.read_csv(filepath)
            print(f"Loaded {key}: {filepath}")
        else:
            print(f"Warning: {filepath} not found")
    
    return (training_stations, distance_data, poi_data, infra_data, 
            census_data, daily_data)

def get_month_year_from_code(month_code):
    """Convert month code (e.g., '202101') to month and year."""
    year = int(month_code[:4])
    month = int(month_code[4:6])
    return month, year

def get_active_stations_for_month(training_stations, month_code):
    """Get list of active stations for a specific month."""
    if month_code not in training_stations.columns:
        return []
    
    # Get stations with value 1 for this month
    active_mask = training_stations[month_code] == 1
    active_stations = training_stations[active_mask]['station_id'].tolist()
    return active_stations

def get_distance_counts_for_station(distance_data, month_code, station_id):
    """Get distance counts for a specific station in a specific month."""
    counts = {}
    
    for distance_key, data in distance_data.items():
        if month_code not in data.columns:
            counts[distance_key] = 0
            continue
            
        # Find the station in the distance data
        station_data = data[data['station_id'] == station_id]
        if len(station_data) > 0:
            # Get the value for this month, handling NaN values
            month_value = station_data[month_code].iloc[0]
            counts[distance_key] = 0 if pd.isna(month_value) else month_value
        else:
            counts[distance_key] = 0
    
    return counts

def get_poi_and_infra_for_station(poi_data, infra_data, station_id):
    """Get POI and infrastructure data for a specific station."""
    # Get POI data for this station
    station_poi = poi_data[poi_data['station_id'] == station_id]
    
    # Get infrastructure data for this station
    station_infra = infra_data[infra_data['station_id'] == station_id]
    
    poi_values = {}
    if len(station_poi) > 0:
        poi_values = {
            'poi_tourism': station_poi['poi_tourism'].iloc[0],
            'poi_education': station_poi['poi_education'].iloc[0],
            'poi_medical': station_poi['poi_medical'].iloc[0],
            'poi_shop': station_poi['poi_shop'].iloc[0],
            'poi_leisure': station_poi['poi_leisure'].iloc[0]
        }
    else:
        poi_values = {
            'poi_tourism': 0, 'poi_education': 0, 'poi_medical': 0,
            'poi_shop': 0, 'poi_leisure': 0
        }
    
    infra_values = {}
    if len(station_infra) > 0:
        infra_values = {
            'bike_route_length_m': station_infra['bike_route_length_m'].iloc[0],
            'street_length_m': station_infra['street_length_m'].iloc[0],
            'rail_stops_count': station_infra['rail_stops_count'].iloc[0],
            'bus_stops_count': station_infra['bus_stops_count'].iloc[0]
        }
    else:
        infra_values = {
            'bike_route_length_m': 0, 'street_length_m': 0,
            'rail_stops_count': 0, 'bus_stops_count': 0
        }
    
    return poi_values, infra_values

def get_census_data_for_station(census_data, station_id):
    """Get census data for a specific station."""
    station_census = census_data[census_data['station_id'] == station_id]
    
    if len(station_census) == 0:
        return {}
    
    # Get all census variables for this station
    census_vars = [
        'pct_white', 'pct_black', 'pct_asian', 'pct_indian', 'pct_hawaiian',
        'pct_two_or_more_races', 'pct_hispanic', 'pct_female', 
        'pct_young_adults_20_34', 'pct_zero_car_ownership', 'unemployment_rate',
        'pct_bachelors_plus', 'pct_drive_alone', 'pct_bike_to_work', 
        'pct_walk_to_work', 'per_capita_income', 'population_density_sq_meter',
        'housing_density_sq_meter'
    ]
    
    census_values = {}
    for var in census_vars:
        if var in station_census.columns:
            census_values[var] = station_census[var].iloc[0]
        else:
            census_values[var] = np.nan
    
    return census_values

def get_daily_trip_data_for_station(daily_data, month_code, station_id):
    """Get daily trip data for a specific station in a specific month."""
    trip_data = {}
    
    for trip_type, data in daily_data.items():
        if month_code not in data.columns:
            trip_data[trip_type] = 0
            continue
            
        # Find the station in the daily data
        station_data = data[data['station_id'] == station_id]
        if len(station_data) > 0:
            month_value = station_data[month_code].iloc[0]
            trip_data[trip_type] = 0 if pd.isna(month_value) else month_value
        else:
            trip_data[trip_type] = 0
    
    return trip_data

def create_dataset_for_period(start_month, end_month, training_stations, 
                             distance_data, poi_data, infra_data, census_data, 
                             daily_data):
    """Create dataset for a specific time period."""
    print(f"\nCreating dataset for period: {start_month} to {end_month}")
    
    # Generate list of months in the period
    months = []
    current_month = start_month
    while current_month <= end_month:
        months.append(str(current_month))
        # Move to next month
        year = int(str(current_month)[:4])
        month = int(str(current_month)[4:6])
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        current_month = int(f"{year:04d}{month:02d}")
    
    print(f"Processing {len(months)} months...")
    
    dataset_rows = []
    total_rows = 0
    
    for month_code in months:
        print(f"Processing month: {month_code}")
        
        # Get active stations for this month
        active_stations = get_active_stations_for_month(training_stations, month_code)
        
        if len(active_stations) == 0:
            print(f"  No active stations for {month_code}")
            continue
        
        print(f"  Found {len(active_stations)} active stations")
        
        # Get month and year
        month, year = get_month_year_from_code(month_code)
        
        # Process each active station
        for station_id in active_stations:
            # Get distance counts for this station
            distance_counts = get_distance_counts_for_station(distance_data, month_code, station_id)
            
            # Get POI and infrastructure data for this station
            poi_values, infra_values = get_poi_and_infra_for_station(poi_data, infra_data, station_id)
            
            # Get census data for this station
            census_values = get_census_data_for_station(census_data, station_id)
            
            # Get daily trip data for this station
            trip_data = get_daily_trip_data_for_station(daily_data, month_code, station_id)
            
            # Create row for this station in this month
            row = {
                'station_id': station_id,
                'month': month,
                'year': year,
                'n250': distance_counts.get('n250', 0),
                'n500': distance_counts.get('n500', 0),
                'n750': distance_counts.get('n750', 0),
                'n1000': distance_counts.get('n1000', 0),
                'n1250': distance_counts.get('n1250', 0),
                'n1500': distance_counts.get('n1500', 0),
                'bike_route_length_m': infra_values.get('bike_route_length_m', 0),
                'street_length_m': infra_values.get('street_length_m', 0),
                'rail_stops_count': infra_values.get('rail_stops_count', 0),
                'bus_stops_count': infra_values.get('bus_stops_count', 0),
                'poi_tourism': poi_values.get('poi_tourism', 0),
                'poi_education': poi_values.get('poi_education', 0),
                'poi_medical': poi_values.get('poi_medical', 0),
                'poi_shop': poi_values.get('poi_shop', 0),
                'poi_leisure': poi_values.get('poi_leisure', 0),
                'pct_white': census_values.get('pct_white', np.nan),
                'pct_black': census_values.get('pct_black', np.nan),
                'pct_asian': census_values.get('pct_asian', np.nan),
                'pct_indian': census_values.get('pct_indian', np.nan),
                'pct_hawaiian': census_values.get('pct_hawaiian', np.nan),
                'pct_two_or_more_races': census_values.get('pct_two_or_more_races', np.nan),
                'pct_hispanic': census_values.get('pct_hispanic', np.nan),
                'pct_female': census_values.get('pct_female', np.nan),
                'pct_young_adults_20_34': census_values.get('pct_young_adults_20_34', np.nan),
                'pct_zero_car_ownership': census_values.get('pct_zero_car_ownership', np.nan),
                'unemployment_rate': census_values.get('unemployment_rate', np.nan),
                'pct_bachelors_plus': census_values.get('pct_bachelors_plus', np.nan),
                'pct_drive_alone': census_values.get('pct_drive_alone', np.nan),
                'pct_bike_to_work': census_values.get('pct_bike_to_work', np.nan),
                'pct_walk_to_work': census_values.get('pct_walk_to_work', np.nan),
                'per_capita_income': census_values.get('per_capita_income', np.nan),
                'population_density_sq_meter': census_values.get('population_density_sq_meter', np.nan),
                'housing_density_sq_meter': census_values.get('housing_density_sq_meter', np.nan),
                'cbike_start': trip_data.get('cbike_start', 0),
                'cbike_end': trip_data.get('cbike_end', 0),
                'ebike_start': trip_data.get('ebike_start', 0),
                'ebike_end': trip_data.get('ebike_end', 0),
                'total_start': trip_data.get('total_start', 0),
                'total_end': trip_data.get('total_end', 0)
            }
            
            dataset_rows.append(row)
        
        total_rows += len(active_stations)
        print(f"  Added {len(active_stations)} rows for {month_code}")
    
    print(f"Total rows created: {total_rows}")
    
    # Create DataFrame
    dataset = pd.DataFrame(dataset_rows)
    
    # Sort by year, month, and station_id
    dataset = dataset.sort_values(['year', 'month', 'station_id'])
    
    return dataset

def main():
    """Main function to create training and test datasets."""
    print("Starting dataset preparation...")
    
    # Load all data files
    (training_stations, distance_data, poi_data, infra_data, 
     census_data, daily_data) = load_data_files()
    
    # Create training dataset (202101 to 202407)
    print("\n" + "="*50)
    training_dataset = create_dataset_for_period(
        202101, 202407, training_stations, distance_data, poi_data, 
        infra_data, census_data, daily_data
    )
    
    # Create test dataset (202408 to 202507)
    print("\n" + "="*50)
    test_dataset = create_dataset_for_period(
        202408, 202507, training_stations, distance_data, poi_data, 
        infra_data, census_data, daily_data
    )
    
    # Save datasets
    print("\n" + "="*50)
    print("Saving datasets...")
    
    # Create result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)
    
    # Save training dataset
    training_file = "result/training_dataset.csv"
    training_dataset.to_csv(training_file, index=False)
    print(f"Training dataset saved to: {training_file}")
    print(f"Training dataset shape: {training_dataset.shape}")
    
    # Save test dataset
    test_file = "result/test_dataset.csv"
    test_dataset.to_csv(test_file, index=False)
    print(f"Test dataset saved to: {test_file}")
    print(f"Test dataset shape: {test_dataset.shape}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    print(f"\nTraining Dataset ({len(training_dataset)} rows):")
    print(f"  Date range: {training_dataset['year'].min()}-{training_dataset['month'].min():02d} to {training_dataset['year'].max()}-{training_dataset['month'].max():02d}")
    print(f"  Unique months: {training_dataset[['year', 'month']].drop_duplicates().shape[0]}")
    print(f"  Unique stations: {training_dataset['station_id'].nunique()}")
    
    print(f"\nTest Dataset ({len(test_dataset)} rows):")
    print(f"  Date range: {test_dataset['year'].min()}-{test_dataset['month'].min():02d} to {test_dataset['year'].max()}-{test_dataset['month'].max():02d}")
    print(f"  Unique months: {test_dataset[['year', 'month']].drop_duplicates().shape[0]}")
    print(f"  Unique stations: {test_dataset['station_id'].nunique()}")
    
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
