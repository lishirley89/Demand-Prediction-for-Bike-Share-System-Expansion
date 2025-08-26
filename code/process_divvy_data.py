#!/usr/bin/env python3
"""
Process Divvy trip data to generate station-level summaries.

For each CSV file in the divvy folder, creates a summary CSV in the trip folder
with station-level statistics including trip counts by rideable type.
"""

import os
import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

def extract_year_month(filename):
    """Extract year and month from filename like '202401-divvy-tripdata.csv'"""
    match = re.match(r'(\d{4})(\d{2})-divvy-tripdata\.csv', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def process_divvy_file(csv_path, output_dir):
    """Process a single Divvy CSV file and generate station summary."""
    
    # Extract year and month from filename
    filename = csv_path.name
    year, month = extract_year_month(filename)
    
    if year is None or month is None:
        print(f"Could not parse year/month from filename: {filename}")
        return None, None
    
    print(f"Processing {filename} (Year: {year}, Month: {month})")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        print(f"  Total trips: {len(df)}")
        
        # Initialize dictionaries to store station data
        stations = {}
        
        # Convert started_at to datetime for date tracking
        df['started_at'] = pd.to_datetime(df['started_at'])
        df['ended_at'] = pd.to_datetime(df['ended_at'])
        
        # Process start stations (keep rows with start station info)
        df_start = df.dropna(subset=['start_station_id', 'start_station_name'])
        print(f"  Trips with start station info: {len(df_start)}")
        
        for _, row in df_start.iterrows():
            station_id = row['start_station_id']
            station_name = row['start_station_name']
            rideable_type = row['rideable_type']
            lat = row['start_lat']
            lng = row['start_lng']
            
            if station_id not in stations:
                stations[station_id] = {
                    'station_id': station_id,
                    'station_name': station_name,
                    'lat': lat,
                    'lng': lng,
                    'ebike_start': 0,
                    'cbike_start': 0,
                    'ebike_end': 0,
                    'cbike_end': 0,
                    'active_dates': set()
                }
            
            if rideable_type == 'electric_bike':
                stations[station_id]['ebike_start'] += 1
            elif rideable_type == 'classic_bike':
                stations[station_id]['cbike_start'] += 1
            
            # Add start date to active dates
            start_date = row['started_at'].date()
            stations[station_id]['active_dates'].add(start_date)
        
        # Process end stations (keep rows with end station info)
        df_end = df.dropna(subset=['end_station_id', 'end_station_name'])
        print(f"  Trips with end station info: {len(df_end)}")
        
        for _, row in df_end.iterrows():
            station_id = row['end_station_id']
            station_name = row['end_station_name']
            rideable_type = row['rideable_type']
            lat = row['end_lat']
            lng = row['end_lng']
            
            if station_id not in stations:
                stations[station_id] = {
                    'station_id': station_id,
                    'station_name': station_name,
                    'lat': lat,
                    'lng': lng,
                    'ebike_start': 0,
                    'cbike_start': 0,
                    'ebike_end': 0,
                    'cbike_end': 0,
                    'active_dates': set()
                }
            
            if rideable_type == 'electric_bike':
                stations[station_id]['ebike_end'] += 1
            elif rideable_type == 'classic_bike':
                stations[station_id]['cbike_end'] += 1
            
            # Add end date to active dates
            end_date = row['ended_at'].date()
            stations[station_id]['active_dates'].add(end_date)
        
        # Convert active_dates sets to counts and create DataFrame
        for station in stations.values():
            station['active_days'] = len(station['active_dates'])
            del station['active_dates']  # Remove set before creating DataFrame
        
        stations_df = pd.DataFrame(list(stations.values()))
        stations_df['month'] = month
        stations_df['year'] = year
        stations_df['total_start'] = stations_df['ebike_start'] + stations_df['cbike_start']
        stations_df['total_end'] = stations_df['ebike_end'] + stations_df['cbike_end']
        
        # Calculate daily averages (avoid division by zero)
        stations_df['daily_ebike_start'] = stations_df['ebike_start'] / stations_df['active_days'].replace(0, 1)
        stations_df['daily_ebike_end'] = stations_df['ebike_end'] / stations_df['active_days'].replace(0, 1)
        stations_df['daily_cbike_start'] = stations_df['cbike_start'] / stations_df['active_days'].replace(0, 1)
        stations_df['daily_cbike_end'] = stations_df['cbike_end'] / stations_df['active_days'].replace(0, 1)
        stations_df['daily_start'] = stations_df['total_start'] / stations_df['active_days'].replace(0, 1)
        stations_df['daily_end'] = stations_df['total_end'] / stations_df['active_days'].replace(0, 1)
        
        # Round daily averages to 2 decimal places
        daily_columns = ['daily_ebike_start', 'daily_ebike_end', 'daily_cbike_start', 
                        'daily_cbike_end', 'daily_start', 'daily_end']
        for col in daily_columns:
            stations_df[col] = stations_df[col].round(2)
        
        # Reorder columns
        columns_order = [
            'month', 'year', 'station_id', 'station_name', 'lat', 'lng',
            'ebike_start', 'ebike_end', 'cbike_start', 'cbike_end',
            'total_start', 'total_end', 'active_days',
            'daily_ebike_start', 'daily_ebike_end', 'daily_cbike_start', 'daily_cbike_end',
            'daily_start', 'daily_end'
        ]
        stations_df = stations_df[columns_order]
        
        # Save to output file
        output_filename = f"{year:04d}{month:02d}-station-summary.csv"
        output_path = output_dir / output_filename
        stations_df.to_csv(output_path, index=False)
        
        print(f"  Generated: {output_filename}")
        print(f"  Stations: {len(stations_df)}")
        print(f"  Total trips: {stations_df['total_start'].sum()} starts, {stations_df['total_end'].sum()} ends")
        
        # Return summary data for monthly aggregation
        summary_data = {
            'year': year,
            'month': month,
            'total_ebike_start': stations_df['ebike_start'].sum(),
            'total_ebike_end': stations_df['ebike_end'].sum(),
            'total_cbike_start': stations_df['cbike_start'].sum(),
            'total_cbike_end': stations_df['cbike_end'].sum(),
            'total_trips_start': stations_df['total_start'].sum(),
            'total_trips_end': stations_df['total_end'].sum()
        }
        
        station_count = {
            'year': year,
            'month': month,
            'active_stations': len(stations_df)
        }
        
        return summary_data, station_count
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, None

def main():
    """Main function to process all Divvy CSV files."""
    
    # Setup paths
    project_root = Path('/Users/shirley/Documents/Research/demand/parkchargebike')
    divvy_dir = project_root / 'divvy'
    trip_dir = project_root / 'trip'
    
    # Create output directory if it doesn't exist
    trip_dir.mkdir(exist_ok=True)
    
    # Find all CSV files in divvy directory
    csv_files = list(divvy_dir.glob('*-divvy-tripdata.csv'))
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Output directory: {trip_dir}")
    print("-" * 50)
    
    # Initialize monthly summary tracking
    monthly_summaries = []
    monthly_stations = []
    
    # Process each file
    for csv_file in sorted(csv_files):
        summary_data, station_count = process_divvy_file(csv_file, trip_dir)
        if summary_data:
            monthly_summaries.append(summary_data)
            monthly_stations.append(station_count)
        print()
    
    # Save monthly summary statistics
    if monthly_summaries:
        monthly_df = pd.DataFrame(monthly_summaries)
        monthly_df = monthly_df.sort_values(['year', 'month'])
        monthly_df.to_csv(trip_dir / 'monthly-trip-summary.csv', index=False)
        print(f"Saved monthly trip summary: {trip_dir / 'monthly-trip-summary.csv'}")
    
    if monthly_stations:
        stations_df = pd.DataFrame(monthly_stations)
        stations_df = stations_df.sort_values(['year', 'month'])
        stations_df.to_csv(trip_dir / 'monthly-active-stations.csv', index=False)
        print(f"Saved monthly active stations: {trip_dir / 'monthly-active-stations.csv'}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
