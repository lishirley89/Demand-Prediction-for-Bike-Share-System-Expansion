#!/usr/bin/env python3
"""
Process NHGIS Census Tract Data for Cook County, Illinois

This script processes the NHGIS dataset nhgis0023_ds267_20235_tract.csv
and filters for Cook County census tracts, calculating various demographic
and socioeconomic variables as requested.

Data source: 2019-2023 American Community Survey 5-Year Estimates
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_nhgis_data(file_path):
    """
    Load NHGIS census tract data from CSV file.
    
    Args:
        file_path (str): Path to the NHGIS CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    print(f"Loading NHGIS data from: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path, low_memory=False)
    
    print(f"Loaded {len(df):,} census tracts")
    print(f"Columns: {len(df.columns)}")
    
    return df

def load_tract_shapefile():
    """
    Load the Illinois census tract shapefile.
    
    Returns:
        geopandas.GeoDataFrame: Shapefile data with geometry
    """
    print("\nLoading tract shapefile...")
    
    try:
        import geopandas as gpd
        shapefile_path = "data/tl_2024_17_tract/tl_2024_17_tract.shp"
        
        if not os.path.exists(shapefile_path):
            print(f"Error: Shapefile not found at {shapefile_path}")
            return None
        
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        print(f"Loaded {len(gdf)} census tracts from shapefile")
        print(f"Shapefile columns: {list(gdf.columns)}")
        print(f"Coordinate system: {gdf.crs}")
        
        return gdf
        
    except ImportError:
        print("Error: geopandas is required to read shapefiles")
        print("Please install with: pip install geopandas")
        return None
    except Exception as e:
        print(f"Error reading shapefile: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def filter_cook_county(df):
    """
    Filter data to keep only Cook County census tracts.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Filtered data for Cook County only
    """
    print("\nFiltering for Cook County...")
    
    # Filter for Cook County
    cook_county = df[df['COUNTY'] == 'Cook County'].copy()
    
    print(f"Found {len(cook_county):,} Cook County census tracts")
    
    return cook_county

def merge_shapefile_with_nhgis(shapefile_gdf, nhgis_df):
    """
    Merge the shapefile data with NHGIS demographic data.
    
    Args:
        shapefile_gdf (geopandas.GeoDataFrame): Shapefile data with geometry
        nhgis_df (pd.DataFrame): NHGIS demographic data
        
    Returns:
        geopandas.GeoDataFrame: Merged data with geometry and demographics
    """
    print("\nMerging shapefile with NHGIS data...")
    
    if shapefile_gdf is None:
        print("No shapefile data available for merging")
        return None
    
    # Create a copy to avoid modifying original
    merged_gdf = shapefile_gdf.copy()
    
    # Check what columns are available for merging
    print(f"Shapefile columns: {list(merged_gdf.columns)}")
    print(f"NHGIS columns: {list(nhgis_df.columns)}")
    
    # Try different merge strategies based on available columns
    merge_successful = False
    
    # Strategy 1: Merge on GEOID
    if 'GEOID' in merged_gdf.columns and 'GEO_ID' in nhgis_df.columns:
        print("Attempting merge on GEOID...")
        # Clean up GEO_ID to match GEOID format
        nhgis_df['GEOID_clean'] = nhgis_df['GEO_ID'].str.replace('1400000US', '')
        
        # Merge on cleaned GEOID
        merged_gdf = merged_gdf.merge(
            nhgis_df, 
            left_on='GEOID', 
            right_on='GEOID_clean', 
            how='inner'
        )
        
        # Drop the temporary column
        merged_gdf = merged_gdf.drop('GEOID_clean', axis=1)
        merge_successful = True
        print(f"Successfully merged on GEOID: {len(merged_gdf)} tracts")
    
    # Strategy 2: Merge on TRACTCE (tract code)
    elif 'TRACTCE' in merged_gdf.columns and 'TRACTA' in nhgis_df.columns:
        print("Attempting merge on tract code...")
        # Clean up tract codes for matching
        merged_gdf['TRACTCE_clean'] = merged_gdf['TRACTCE'].astype(str).str.zfill(6)
        nhgis_df['TRACTA_clean'] = nhgis_df['TRACTA'].astype(str).str.zfill(6)
        
        # Merge on tract codes
        merged_gdf = merged_gdf.merge(
            nhgis_df, 
            left_on='TRACTCE_clean', 
            right_on='TRACTA_clean', 
            how='inner'
        )
        
        # Drop temporary columns
        merged_gdf = merged_gdf.drop(['TRACTCE_clean', 'TRACTA_clean'], axis=1)
        merge_successful = True
        print(f"Successfully merged on tract code: {len(merged_gdf)} tracts")
    
    # Strategy 3: Merge on county and tract combination
    elif 'COUNTYFP' in merged_gdf.columns and 'TRACTA' in nhgis_df.columns:
        print("Attempting merge on county + tract combination...")
        # Create composite key
        merged_gdf['COUNTY_TRACT'] = merged_gdf['COUNTYFP'].astype(str) + merged_gdf['TRACTCE'].astype(str).str.zfill(6)
        nhgis_df['COUNTY_TRACT'] = nhgis_df['COUNTYA'].astype(str) + nhgis_df['TRACTA'].astype(str).str.zfill(6)
        
        # Merge on composite key
        merged_gdf = merged_gdf.merge(
            nhgis_df, 
            left_on='COUNTY_TRACT', 
            right_on='COUNTY_TRACT', 
            how='inner'
        )
        
        # Drop temporary column
        merged_gdf = merged_gdf.drop('COUNTY_TRACT', axis=1)
        merge_successful = True
        print(f"Successfully merged on county+tract: {len(merged_gdf)} tracts")
    
    if not merge_successful:
        print("Warning: Could not merge shapefile with NHGIS data")
        print("Available columns for merging:")
        print(f"Shapefile: {[col for col in merged_gdf.columns if col in ['GEOID', 'TRACTCE', 'COUNTYFP', 'NAME']]}")
        print(f"NHGIS: {[col for col in nhgis_df.columns if col in ['GEO_ID', 'TRACTA', 'COUNTYA', 'NAME_E']]}")
        return None
    
    # Filter for Cook County only
    if 'COUNTY' in merged_gdf.columns:
        merged_gdf = merged_gdf[merged_gdf['COUNTY'] == 'Cook County'].copy()
        print(f"Filtered to {len(merged_gdf)} Cook County tracts")
    
    return merged_gdf

def calculate_requested_variables(df):
    """
    Calculate the specific variables requested by the user.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with calculated variables
    """
    print("\nCalculating requested variables...")
    
    # Create a copy to avoid modifying original
    result = df.copy()
    
    # a. % white
    result['pct_white'] = (result['ASN2E002'] / result['ASN2E001'] * 100).round(2)
    
    # b. % black
    result['pct_black'] = (result['ASN2E003'] / result['ASN2E001'] * 100).round(2)
    
    # c. % asian
    result['pct_asian'] = (result['ASN2E005'] / result['ASN2E001'] * 100).round(2)
    
    # d. % indian (American Indian and Alaska Native)
    result['pct_indian'] = (result['ASN2E004'] / result['ASN2E001'] * 100).round(2)
    
    # e. % hawaiian (Native Hawaiian and Other Pacific Islander)
    result['pct_hawaiian'] = (result['ASN2E006'] / result['ASN2E001'] * 100).round(2)
    
    # f. % two or more races
    result['pct_two_or_more_races'] = (result['ASN2E008'] / result['ASN2E001'] * 100).round(2)
    
    # g. % zero-car ownership (no vehicles available)
    result['pct_zero_car_ownership'] = ((result['ASUTE003'] + result['ASUTE010']) / result['ASUTE001'] * 100).round(2)
    
    # h. % young adults, age 20-34
    # Male: 20, 21, 22-24, 25-29, 30-34 (ASNQE008, ASNQE009, ASNQE010, ASNQE011, ASNQE012)
    # Female: 20, 21, 22-24, 25-29, 30-34 (ASNQE032, ASNQE033, ASNQE034, ASNQE035, ASNQE036)
    result['pct_young_adults_20_34'] = ((result['ASNQE008'] + result['ASNQE009'] + result['ASNQE010'] + 
                                         result['ASNQE011'] + result['ASNQE012'] + 
                                         result['ASNQE032'] + result['ASNQE033'] + result['ASNQE034'] + 
                                         result['ASNQE035'] + result['ASNQE036']) / result['ASNQE001'] * 100).round(2)
    
    # i. Unemployment rate: # unemployed/# in labor force
    result['unemployment_rate'] = (result['ASSRE005'] / result['ASSRE002'] * 100).round(2)
    
    # j. Population density (using land area from shapefile)
    if 'ALAND' in result.columns:
        # Keep ALAND in square meters and calculate density per square meter
        result['land_area_sq_meters'] = result['ALAND']
        result['population_density_sq_meter'] = (result['ASN1E001'] / result['ALAND']).round(6)
        print("Calculated population density per square meter using shapefile land area")
    else:
        # Fallback to population per tract
        result['population_density_sq_meter'] = np.nan
        result['land_area_sq_meters'] = 1.0
        print("Using population per tract as density proxy (land area not available)")
    
    # k. % female
    # Female total is ASNQE026, but we need to calculate from individual age groups
    # Sum all female age groups: ASNQE027 through ASNQE049
    female_cols = [col for col in result.columns if col.startswith('ASNQE') and 
                   col.endswith(('27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'))]
    result['pct_female'] = (result[female_cols].sum(axis=1) / result['ASNQE001'] * 100).round(2)
    
    # l. % hispanic
    result['pct_hispanic'] = (result['ASOBE003'] / result['ASOBE001'] * 100).round(2)
    
    # n. Education: % population aged 25 and above with bachelor or higher
    result['pct_bachelors_plus'] = ((result['ASP3E021'] + result['ASP3E022'] + 
                                     result['ASP3E023'] + result['ASP3E024'] + result['ASP3E025']) / result['ASP3E001'] * 100).round(2)
    
    # o. % drive alone to work, ACS
    result['pct_drive_alone'] = (result['ASORE003'] / result['ASORE001'] * 100).round(2)
    
    # % bike to work, ACS
    result['pct_bike_to_work'] = (result['ASORE018'] / result['ASORE001'] * 100).round(2)
    
    # % walk to work, ACS
    result['pct_walk_to_work'] = (result['ASORE019'] / result['ASORE001'] * 100).round(2)
    
    # p. Housing density, ACS (housing units per tract and per square meter)
    result['housing_density'] = result['ASS7E001']
    if 'land_area_sq_meters' in result.columns:
        result['housing_density_sq_meter'] = (result['ASS7E001'] / result['land_area_sq_meters']).round(6)
    
    # Additional useful variables
    # Total population
    result['total_population'] = result['ASN1E001']
    
    # Per capita income (available in this dataset)
    result['per_capita_income'] = result['ASRTE001']
    
    # Employment rate
    result['employment_rate'] = (result['ASSRE004'] / result['ASSRE001'] * 100).round(2)
    
    # Labor force participation rate
    result['labor_force_participation_rate'] = (result['ASSRE002'] / result['ASSRE001'] * 100).round(2)
    
    # Public transit percentage
    result['pct_public_transit'] = (result['ASORE010'] / result['ASORE001'] * 100).round(2)
    
    # Work from home percentage
    result['pct_work_from_home'] = (result['ASORE021'] / result['ASORE001'] * 100).round(2)
    
    # Vehicle ownership breakdown
    result['pct_one_car'] = ((result['ASUTE004'] + result['ASUTE011']) / result['ASUTE001'] * 100).round(2)
    result['pct_two_plus_cars'] = ((result['ASUTE005'] + result['ASUTE006'] + result['ASUTE007'] + result['ASUTE008'] + 
                                    result['ASUTE012'] + result['ASUTE013'] + result['ASUTE014'] + result['ASUTE015']) / result['ASUTE001'] * 100).round(2)
    
    # Housing tenure
    result['pct_owner_occupied'] = (result['ASUTE002'] / result['ASUTE001'] * 100).round(2)
    result['pct_renter_occupied'] = (result['ASUTE009'] / result['ASUTE001'] * 100).round(2)
    
    print("All requested variables calculated successfully")
    
    return result

def create_summary_statistics(df):
    """
    Create summary statistics for the calculated variables.
    
    Args:
        df (pd.DataFrame): Input dataframe with calculated variables
        
    Returns:
        pd.DataFrame: Summary statistics
    """
    print("\nCreating summary statistics...")
    
    # Select only the calculated demographic variables
    demographic_cols = [col for col in df.columns if col.startswith('pct_') or 
                       col in ['total_population', 'population_density_sq_meter', 'land_area_sq_meters',
                               'housing_density', 'housing_density_sq_meter', 'per_capita_income', 
                               'unemployment_rate', 'employment_rate', 'labor_force_participation_rate']]
    
    # Filter out columns that don't exist
    demographic_cols = [col for col in demographic_cols if col in df.columns]
    
    # Create summary statistics
    summary = df[demographic_cols].describe()
    
    # Add count of non-null values
    summary.loc['count'] = df[demographic_cols].count()
    
    # Add count of census tracts
    summary.loc['census_tracts'] = len(df)
    
    return summary

def save_results(gdf, summary, output_dir):
    """
    Save the processed data and summary statistics.
    
    Args:
        gdf (geopandas.GeoDataFrame): Processed geodataframe with geometry
        summary (pd.DataFrame): Summary statistics
        output_dir (str): Output directory path
    """
    print(f"\nSaving results to: {output_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define key variables for the clean GeoJSON
    key_vars = ['GEOID', 'TRACTCE', 'NAME', 'total_population', 
                'pct_white', 'pct_black', 'pct_asian', 'pct_indian', 'pct_hawaiian', 'pct_two_or_more_races',
                'pct_hispanic', 'pct_female', 'pct_young_adults_20_34', 'pct_zero_car_ownership',
                'unemployment_rate', 'pct_bachelors_plus', 'pct_drive_alone', 'pct_bike_to_work', 
                'pct_walk_to_work', 'housing_density', 'per_capita_income']
    
    # Add land area and density columns if available
    if 'land_area_sq_meters' in gdf.columns:
        key_vars.extend(['land_area_sq_meters'])
    if 'population_density_sq_meter' in gdf.columns:
        key_vars.extend(['population_density_sq_meter'])
    if 'housing_density_sq_meter' in gdf.columns:
        key_vars.extend(['housing_density_sq_meter'])
    
    # Filter to only include columns that exist
    key_vars = [col for col in key_vars if col in gdf.columns]
    
    # Create clean GeoJSON with only key variables and geometry
    clean_gdf = gdf[['geometry'] + key_vars].copy()
    
    # Save as GeoJSON (main output) - clean version with only key variables
    geojson_file = os.path.join(output_dir, 'cook_county_census_tracts.geojson')
    clean_gdf.to_file(geojson_file, driver='GeoJSON')
    print(f"Clean GeoJSON file saved to: {geojson_file} (geometry + key variables only)")
    
    # Save full processed data as CSV (without geometry)
    csv_file = os.path.join(output_dir, 'cook_county_census_tracts_processed.csv')
    gdf.drop(columns=['geometry']).to_csv(csv_file, index=False)
    print(f"Full CSV file saved to: {csv_file}")
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, 'cook_county_summary_statistics.csv')
    summary.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Save a simplified version with key variables (CSV without geometry)
    simplified_file = os.path.join(output_dir, 'cook_county_key_variables.csv')
    clean_gdf.drop(columns=['geometry']).to_csv(simplified_file, index=False)
    print(f"Key variables CSV saved to: {simplified_file}")
    
    print(f"\nGeoJSON contains {len(key_vars)} key variables + geometry")
    print(f"Key variables: {', '.join(key_vars)}")

def main():
    """
    Main function to process NHGIS data for Cook County.
    """
    print("=" * 80)
    print("NHGIS Cook County Census Tract Data Processing with Shapefile Integration")
    print("=" * 80)
    
    # File paths
    data_dir = "data/nhgis0024_csv"
    input_file = os.path.join(data_dir, "nhgis0024_ds267_20235_tract.csv")
    output_dir = "data"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    try:
        # Load NHGIS data
        print("\nStep 1: Loading NHGIS data...")
        nhgis_df = load_nhgis_data(input_file)
        
        # Load tract shapefile
        print("\nStep 2: Loading tract shapefile...")
        shapefile_gdf = load_tract_shapefile()
        
        if shapefile_gdf is None:
            print("Error: Could not load shapefile. Exiting.")
            return
        
        # Filter NHGIS data for Cook County
        print("\nStep 3: Filtering for Cook County...")
        cook_county_nhgis = filter_cook_county(nhgis_df)
        
        if len(cook_county_nhgis) == 0:
            print("No Cook County data found in NHGIS!")
            return
        
        # Merge shapefile with NHGIS data
        print("\nStep 4: Merging shapefile with NHGIS data...")
        merged_gdf = merge_shapefile_with_nhgis(shapefile_gdf, cook_county_nhgis)
        
        if merged_gdf is None:
            print("Error: Could not merge shapefile with NHGIS data. Exiting.")
            return
        
        # Calculate requested variables
        print("\nStep 5: Calculating demographic variables...")
        processed_gdf = calculate_requested_variables(merged_gdf)
        
        # Create summary statistics
        print("\nStep 6: Creating summary statistics...")
        summary_stats = create_summary_statistics(processed_gdf)
        
        # Save results
        print("\nStep 7: Saving results...")
        save_results(processed_gdf, summary_stats, output_dir)
        
        print("\n" + "=" * 80)
        print("Processing completed successfully!")
        print("=" * 80)
        
        # Display some key statistics
        print(f"\nCook County Census Tracts: {len(processed_gdf):,}")
        print(f"Total Population: {processed_gdf['total_population'].sum():,}")
        print(f"Average Population per Tract: {processed_gdf['total_population'].mean():.0f}")
        print(f"Median Population per Tract: {processed_gdf['total_population'].median():.0f}")
        
        # Display land area information if available
        if 'land_area_sq_meters' in processed_gdf.columns and processed_gdf['land_area_sq_meters'].notna().any():
            print(f"\nLand Area Information:")
            print(f"Average Land Area per Tract: {processed_gdf['land_area_sq_meters'].mean():.2f} sq meters")
            print(f"Total Land Area: {processed_gdf['land_area_sq_meters'].sum():.2f} sq meters")
            
            if 'population_density_sq_meter' in processed_gdf.columns:
                print(f"Average Population Density: {processed_gdf['population_density_sq_meter'].mean():.0f} people/sq meter")
        
        # Display some key percentages
        print(f"\nKey Demographics (averages across tracts):")
        print(f"White: {processed_gdf['pct_white'].mean():.1f}%")
        print(f"Black: {processed_gdf['pct_black'].mean():.1f}%")
        print(f"Hispanic: {processed_gdf['pct_hispanic'].mean():.1f}%")
        print(f"Asian: {processed_gdf['pct_asian'].mean():.1f}%")
        print(f"Zero-car ownership: {processed_gdf['pct_zero_car_ownership'].mean():.1f}%")
        print(f"Bike to work: {processed_gdf['pct_bike_to_work'].mean():.1f}%")
        print(f"Walk to work: {processed_gdf['pct_walk_to_work'].mean():.1f}%")
        
        print(f"\nOutput files:")
        print(f"- GeoJSON: data/cook_county_census_tracts.geojson (geometry + key variables only)")
        print(f"- CSV: data/cook_county_census_tracts_processed.csv (without geometry)")
        print(f"- Summary: data/cook_county_summary_statistics.csv")
        print(f"- Key variables: data/cook_county_key_variables.csv")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set output directory
    output_dir = "data"
    
    # Run the main processing
    main()
    
    print(f"\nOutput files:")
    print(f"- GeoJSON: data/cook_county_census_tracts.geojson (geometry + key variables only)")
    print(f"- CSV: data/cook_county_census_tracts_processed.csv (without geometry)")
    print(f"- Summary: data/cook_county_summary_statistics.csv")
    print(f"- Key Variables: data/cook_county_key_variables.csv")
