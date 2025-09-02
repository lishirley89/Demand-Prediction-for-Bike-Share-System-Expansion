import pandas as pd
import re

def create_training_dataset(input_file, output_file):
    """
    Create training dataset by removing test and repair stations from master_stations.csv
    
    Args:
        input_file (str): Path to input master_stations.csv file
        output_file (str): Path to output master_stations_fortraining.csv file
    """
    print("Loading master_stations.csv...")
    df = pd.read_csv(input_file)
    
    initial_count = len(df)
    print(f"Initial number of stations: {initial_count}")
    
    # Remove stations with 'test' in either name or ID (case insensitive)
    print("\nRemoving stations with 'test' in either name or ID...")
    test_mask = (
        df['station_name'].str.contains('test', case=False, na=False) |
        df['station_id'].astype(str).str.contains('test', case=False, na=False)
    )
    test_count = test_mask.sum()
    df_no_test = df[~test_mask]
    print(f"Removed {test_count} stations containing 'test' in either name or ID")
    print(f"Stations remaining after removing 'test': {len(df_no_test)}")
    
    # Remove stations with 'repair' in either name or ID (case insensitive)
    print("\nRemoving stations with 'repair' in either name or ID...")
    repair_mask = (
        df_no_test['station_name'].str.contains('repair', case=False, na=False) |
        df_no_test['station_id'].astype(str).str.contains('repair', case=False, na=False)
    )
    repair_count = repair_mask.sum()
    df_final = df_no_test[~repair_mask]
    print(f"Removed {repair_count} stations containing 'repair' in either name or ID")
    print(f"Stations remaining after removing 'repair': {len(df_final)}")
    
    # Summary of removals
    total_removed = initial_count - len(df_final)
    print(f"\nTotal stations removed: {total_removed}")
    print(f"Final training dataset size: {len(df_final)}")
    
    # Save the filtered dataset
    print(f"\nSaving training dataset to: {output_file}")
    df_final.to_csv(output_file, index=False)
    print("Training dataset created successfully!")
    
    return df_final

def main():
    # Define file paths
    input_file = "result/master_stations.csv"
    output_file = "result/master_stations_fortraining.csv"
    
    # Create the result directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create the training dataset
    training_df = create_training_dataset(input_file, output_file)
    
    # Display first few rows of the training dataset
    print("\nFirst 5 rows of training dataset:")
    print(training_df[['station_id', 'station_name']].head())

if __name__ == "__main__":
    main()
