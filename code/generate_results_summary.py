#!/usr/bin/env python3
"""
Generate a summary of all available results and files in the project.
"""

import os
import pandas as pd
from pathlib import Path

def generate_summary():
    """Generate a comprehensive summary of project results."""
    
    result_dir = Path('result')
    if not result_dir.exists():
        print("Result directory not found!")
        return
    
    print("="*60)
    print("BIKE SHARE DEMAND PREDICTION - RESULTS SUMMARY")
    print("="*60)
    
    # Model Performance Summary
    print("\n📊 MODEL PERFORMANCE SUMMARY")
    print("-" * 40)
    
    if (result_dir / 'xgboost_results.csv').exists():
        xgb_results = pd.read_csv(result_dir / 'xgboost_results.csv')
        print("\nXGBoost Models:")
        for _, row in xgb_results.iterrows():
            target = row['target_variable'].replace('_', ' ').title()
            print(f"  {target}:")
            print(f"    Test R²: {row['test_r2']:.3f}")
            print(f"    Test RMSE: {row['test_rmse']:.3f}")
            print(f"    Test MAE: {row['test_mae']:.3f}")
    
    if (result_dir / 'best_models_summary.csv').exists():
        lr_results = pd.read_csv(result_dir / 'best_models_summary.csv')
        print("\nLinear Regression Models:")
        for _, row in lr_results.iterrows():
            target = row['target_variable'].replace('_', ' ').title()
            print(f"  {target}:")
            print(f"    Test R²: {row['test_r2']:.3f}")
            print(f"    Test RMSE: {row['test_rmse']:.3f}")
            print(f"    Test MAE: {row['test_mae']:.3f}")
    
    # Data Summary
    print("\n📈 DATA SUMMARY")
    print("-" * 40)
    
    if (result_dir / 'station_analysis_summary.csv').exists():
        station_summary = pd.read_csv(result_dir / 'station_analysis_summary.csv')
        print("\nStation Statistics:")
        for _, row in station_summary.iterrows():
            print(f"  {row['metric']}: {row['count']:,}")
    
    if (result_dir / 'monthly-active-stations.csv').exists():
        monthly_data = pd.read_csv(result_dir / 'monthly-active-stations.csv')
        print(f"\nTime Period: {monthly_data['year'].min()}-{monthly_data['month'].min():02d} to {monthly_data['year'].max()}-{monthly_data['month'].max():02d}")
        print(f"Total months: {len(monthly_data)}")
        print(f"Average active stations: {monthly_data['active_stations'].mean():.0f}")
        print(f"Peak active stations: {monthly_data['active_stations'].max()}")
    
    # Available Files
    print("\n📁 AVAILABLE FILES")
    print("-" * 40)
    
    file_categories = {
        'Model Results': ['best_models_summary.csv', 'xgboost_results.csv', 'feature_importance.csv', 'linear_regression_results.txt'],
        'Trained Models': [f for f in result_dir.glob('*.joblib')],
        'Error Analysis': [f for f in result_dir.glob('instance_errors_*.csv')] + [f for f in result_dir.glob('error_heatmap_*.html')],
        'SHAP Analysis': [f for f in result_dir.glob('shap_*.csv')],
        'Data Files': [f for f in result_dir.glob('master_stations_*.csv')] + [f for f in result_dir.glob('*dataset*.csv')],
        'Visualizations': [f for f in result_dir.glob('*.png')],
        'Other': [f for f in result_dir.glob('*.geojson')] + [f for f in result_dir.glob('*.txt')]
    }
    
    for category, files in file_categories.items():
        if files:
            print(f"\n{category}:")
            for file in sorted(files):
                if isinstance(file, str):
                    file_path = result_dir / file
                    if file_path.exists():
                        size = file_path.stat().st_size
                        print(f"  ✓ {file} ({size:,} bytes)")
                else:
                    size = file.stat().st_size
                    print(f"  ✓ {file.name} ({size:,} bytes)")
    
    # Top Features
    print("\n🔍 TOP PREDICTORS")
    print("-" * 40)
    
    if (result_dir / 'feature_importance.csv').exists():
        feature_importance = pd.read_csv(result_dir / 'feature_importance.csv')
        
        for target in ['cbike_start', 'cbike_end', 'ebike_start', 'ebike_end']:
            target_data = feature_importance[feature_importance['target_variable'] == target].head(10)
            print(f"\n{target.replace('_', ' ').title()} - Top 10 Features:")
            for _, row in target_data.iterrows():
                print(f"  {row['rank']:2d}. {row['feature']:<25} {row['coefficient']:>8.3f}")
    
    print("\n" + "="*60)
    print("SUMMARY COMPLETE")
    print("="*60)

if __name__ == "__main__":
    generate_summary()
