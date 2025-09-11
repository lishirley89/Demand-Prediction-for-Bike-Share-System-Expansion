#!/usr/bin/env python3
"""
Create summary plots for the README documentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_model_performance_plot():
    """Create a bar chart comparing model performance."""
    # Load results
    lr_results = pd.read_csv('result/best_models_summary.csv')
    xgb_results = pd.read_csv('result/xgboost_results.csv')
    
    # Prepare data
    models = ['Linear Regression', 'XGBoost']
    targets = ['cbike_start', 'cbike_end', 'ebike_start', 'ebike_end']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    metrics = ['test_r2', 'test_rmse', 'test_mae']
    metric_names = ['R² Score', 'RMSE', 'MAE']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Get data for each target
        lr_values = []
        xgb_values = []
        
        for target in targets:
            lr_val = lr_results[lr_results['target_variable'] == target][metric].iloc[0]
            xgb_val = xgb_results[xgb_results['target_variable'] == target][metric].iloc[0]
            lr_values.append(lr_val)
            xgb_values.append(xgb_val)
        
        x = np.arange(len(targets))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lr_values, width, label='Linear Regression', alpha=0.8)
        bars2 = ax.bar(x + width/2, xgb_values, width, label='XGBoost', alpha=0.8)
        
        ax.set_xlabel('Target Variable')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Model Performance: {metric_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('result/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_plot():
    """Create a plot showing top feature importance."""
    # Load feature importance data
    feature_importance = pd.read_csv('result/feature_importance.csv')
    
    # Get top 15 features for each target
    targets = ['cbike_start', 'cbike_end', 'ebike_start', 'ebike_end']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, target in enumerate(targets):
        ax = axes[i]
        
        # Get top 15 features for this target
        target_data = feature_importance[feature_importance['target_variable'] == target].head(15)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(target_data))
        ax.barh(y_pos, target_data['coefficient'], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(target_data['feature'], fontsize=10)
        ax.set_xlabel('Coefficient Value')
        ax.set_title(f'Top 15 Features: {target.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for j, v in enumerate(target_data['coefficient']):
            ax.text(v + 0.01 if v >= 0 else v - 0.01, j, f'{v:.2f}', 
                   va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('result/feature_importance_top15.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_monthly_trends_plot():
    """Create a plot showing monthly trends in active stations."""
    # Load monthly data
    monthly_data = pd.read_csv('result/monthly-active-stations.csv')
    
    # Create datetime column
    monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
    
    plt.figure(figsize=(14, 8))
    plt.plot(monthly_data['date'], monthly_data['active_stations'], 
             marker='o', linewidth=2, markersize=4, color='steelblue', alpha=0.8)
    
    plt.title('Number of Active Stations by Month (2021-2025)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Active Stations', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(monthly_data)), monthly_data['active_stations'], 1)
    p = np.poly1d(z)
    plt.plot(monthly_data['date'], p(range(len(monthly_data))), 
             "--", color='red', alpha=0.7, linewidth=2, label=f'Trend (slope: {z[0]:.1f})')
    
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('result/monthly_active_stations_trend.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_error_distribution_plot():
    """Create plots showing error distributions."""
    # Load instance error data
    error_files = [
        'result/instance_errors_cbike_start.csv',
        'result/instance_errors_cbike_end.csv', 
        'result/instance_errors_ebike_start.csv',
        'result/instance_errors_ebike_end.csv'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, error_file in enumerate(error_files):
        if not Path(error_file).exists():
            continue
            
        ax = axes[i]
        df = pd.read_csv(error_file)
        
        # Create histogram of absolute errors
        ax.hist(df['abs_error'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution: {error_file.split("_")[-1].replace(".csv", "").replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = df['abs_error'].mean()
        median_error = df['abs_error'].median()
        ax.axvline(mean_error, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_error:.2f}')
        ax.axvline(median_error, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_error:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('result/error_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Create all summary plots."""
    print("Creating summary plots for README...")
    
    # Ensure result directory exists
    Path('result').mkdir(exist_ok=True)
    
    try:
        print("1. Creating model performance comparison...")
        create_model_performance_plot()
        
        print("2. Creating feature importance plots...")
        create_feature_importance_plot()
        
        print("3. Creating monthly trends plot...")
        create_monthly_trends_plot()
        
        print("4. Creating error distribution plots...")
        create_error_distribution_plot()
        
        print("All plots created successfully!")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        print("Some plots may not be available if required data files are missing.")

if __name__ == "__main__":
    main()
