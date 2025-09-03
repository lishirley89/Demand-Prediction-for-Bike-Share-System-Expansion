#!/usr/bin/env python3
"""
Linear Regression Modeling Script for Bike Station Demand Analysis.

This script:
1. Loads preprocessed training and test datasets
2. Creates separate models for cbike_start, cbike_end, ebike_start, ebike_end
3. Performs VIF analysis to remove high collinearity variables
4. Applies log transformation to skewed variables
5. Splits training data into 80% training and 20% validation
6. Performs 5-fold cross-validation for robust performance estimation
7. Evaluates models using R², RMSE, and MAE
8. Selects best performing model and applies to test data
9. Records comprehensive results

Outputs:
- result/linear_regression_results.txt - Comprehensive modeling results
- result/best_models_summary.csv - Summary of best performing models
- result/feature_importance.csv - Feature importance for each model
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAIN_PATH = "result/training_dataset_preprocessed.csv"
TEST_PATH = "result/test_dataset_preprocessed.csv"
OUT_DIR = "result"

# Dependent variables to model
DEPENDENT_VARS = ["cbike_start", "cbike_end", "ebike_start", "ebike_end"]

# Columns to exclude from features
EXCLUDE_COLS = ["station_id", "total_start", "total_end", "year"]

# VIF threshold for removing collinear variables
VIF_THRESHOLD = 10.0

# Skewness threshold for log transformation
SKEW_THRESHOLD = 1.0

# Cross-validation settings
CV_FOLDS = 5

def ensure_out_dir():
    """Ensure output directory exists."""
    os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    """Load training and test datasets."""
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df

def prepare_features(train_df, test_df):
    """Prepare feature columns excluding specified columns."""
    # Get all columns except excluded ones and dependent variables
    feature_cols = [col for col in train_df.columns 
                   if col not in EXCLUDE_COLS + DEPENDENT_VARS]
    
    print(f"Feature columns ({len(feature_cols)}): {', '.join(feature_cols)}")
    
    return feature_cols

def check_skewness(df, feature_cols):
    """Check skewness of features and identify variables for log transformation."""
    skewed_features = []
    
    for col in feature_cols:
        if col.startswith('month_'):  # Skip month one-hot columns
            continue
            
        skewness = df[col].skew()
        if abs(skewness) > SKEW_THRESHOLD:
            skewed_features.append((col, skewness))
    
    print(f"\nSkewed features (threshold: {SKEW_THRESHOLD}):")
    for col, skew in skewed_features:
        print(f"  {col}: {skew:.3f}")
    
    return [col for col, _ in skewed_features]

def apply_log_transformation(df, skewed_features):
    """Apply log transformation to skewed features."""
    df_transformed = df.copy()
    
    for col in skewed_features:
        # Add small constant to avoid log(0)
        min_val = df[col].min()
        if min_val <= 0:
            constant = abs(min_val) + 1e-6
        else:
            constant = 0
            
        df_transformed[f'log_{col}'] = np.log(df[col] + constant)
        print(f"  Applied log transformation to {col}")
    
    return df_transformed

def calculate_vif(X):
    """Calculate VIF for all features."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

def remove_high_vif_features(X, threshold=VIF_THRESHOLD):
    """Remove features with high VIF iteratively."""
    print(f"\nRemoving features with VIF > {threshold}...")
    
    while True:
        vif_data = calculate_vif(X)
        max_vif = vif_data['VIF'].max()
        
        if max_vif <= threshold:
            break
            
        # Remove feature with highest VIF
        feature_to_remove = vif_data.iloc[0]['Variable']
        X = X.drop(columns=[feature_to_remove])
        print(f"  Removed {feature_to_remove} (VIF: {max_vif:.2f})")
    
    print(f"Final feature count after VIF removal: {X.shape[1]}")
    return X

def perform_cross_validation(X, y, model_name):
    """Perform 5-fold cross-validation and return scores."""
    print(f"  Performing {CV_FOLDS}-fold cross-validation...")
    
    # Initialize model
    model = LinearRegression()
    
    # Setup cross-validation
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    
    # Perform cross-validation for different metrics
    cv_r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    cv_rmse_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
    cv_mae_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    
    # Convert negative scores to positive for RMSE and MAE
    cv_rmse_scores = -cv_rmse_scores
    cv_mae_scores = -cv_mae_scores
    
    # Calculate statistics
    cv_results = {
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        'cv_rmse_mean': cv_rmse_scores.mean(),
        'cv_rmse_std': cv_rmse_scores.std(),
        'cv_mae_mean': cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std(),
        'cv_r2_scores': cv_r2_scores,
        'cv_rmse_scores': cv_rmse_scores,
        'cv_mae_scores': cv_mae_scores
    }
    
    print(f"    CV R²: {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
    print(f"    CV RMSE: {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f}")
    print(f"    CV MAE: {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f}")
    
    return cv_results

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_full, y_full, model_name):
    """Train model and evaluate on validation set and full training data with cross-validation."""
    # Train model on training split
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions on training and validation splits
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics on splits
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    # Perform cross-validation on full training data
    cv_results = perform_cross_validation(X_full, y_full, model_name)
    
    # Feature importance (absolute coefficients)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': np.abs(model.coef_)
    }).sort_values('coefficient', ascending=False)
    
    results = {
        'model_name': model_name,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'cv_r2_mean': cv_results['cv_r2_mean'],
        'cv_r2_std': cv_results['cv_r2_std'],
        'cv_rmse_mean': cv_results['cv_rmse_mean'],
        'cv_rmse_std': cv_results['cv_rmse_std'],
        'cv_mae_mean': cv_results['cv_mae_mean'],
        'cv_mae_std': cv_results['cv_mae_std'],
        'feature_importance': feature_importance,
        'model': model,
        'feature_names': X_train.columns.tolist()
    }
    
    return results

def evaluate_on_test_data(model, X_test, y_test, feature_names):
    """Evaluate model on test data."""
    # Ensure test data has same features as training
    X_test_aligned = X_test[feature_names]
    
    # Predictions
    y_test_pred = model.predict(X_test_aligned)
    
    # Calculate metrics
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    return {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }

def main():
    """Main execution function."""
    ensure_out_dir()
    
    # Load data
    train_df, test_df = load_data()
    
    # Prepare features
    feature_cols = prepare_features(train_df, test_df)
    
    # Check for skewed features
    skewed_features = check_skewness(train_df, feature_cols)
    
    # Apply log transformation
    if skewed_features:
        print(f"\nApplying log transformation to {len(skewed_features)} features...")
        train_df_transformed = apply_log_transformation(train_df, skewed_features)
        test_df_transformed = apply_log_transformation(test_df, skewed_features)
        
        # Update feature columns to include log-transformed features
        log_features = [f'log_{col}' for col in skewed_features]
        feature_cols.extend(log_features)
    else:
        train_df_transformed = train_df.copy()
        test_df_transformed = test_df.copy()
    
    # Prepare feature matrix
    X_train = train_df_transformed[feature_cols]
    X_test = test_df_transformed[feature_cols]
    
    # Remove high VIF features
    X_train_clean = remove_high_vif_features(X_train.copy())
    final_features = X_train_clean.columns.tolist()
    
    # Align test data with final features
    X_test_clean = test_df_transformed[final_features]
    
    print(f"\nFinal feature set ({len(final_features)}): {', '.join(final_features)}")
    
    # Results storage
    all_results = []
    best_models = []
    
    # Train models for each dependent variable
    for target in DEPENDENT_VARS:
        print(f"\n{'='*60}")
        print(f"MODELING: {target}")
        print(f"{'='*60}")
        
        # Prepare target variable
        y_train = train_df_transformed[target]
        y_test = test_df_transformed[target]
        
        # Split training data (80% train, 20% validation)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_clean, y_train, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {X_train_split.shape[0]}")
        print(f"Validation set size: {X_val_split.shape[0]}")
        
        # Train and evaluate model
        results = train_and_evaluate_model(
            X_train_split, y_train_split, 
            X_val_split, y_val_split,
            X_train_clean, y_train,  # Full training data for CV
            target
        )
        
        # Evaluate on test data
        test_results = evaluate_on_test_data(
            results['model'], X_test_clean, y_test, final_features
        )
        
        # Store results
        model_summary = {
            'target_variable': target,
            'train_r2': results['train_r2'],
            'val_r2': results['val_r2'],
            'test_r2': test_results['test_r2'],
            'train_rmse': results['train_rmse'],
            'val_rmse': results['val_rmse'],
            'test_rmse': test_results['test_rmse'],
            'train_mae': results['train_mae'],
            'val_mae': results['val_mae'],
            'test_mae': test_results['test_mae'],
            'cv_r2_mean': results['cv_r2_mean'],
            'cv_r2_std': results['cv_r2_std'],
            'cv_rmse_mean': results['cv_rmse_mean'],
            'cv_rmse_std': results['cv_rmse_std'],
            'cv_mae_mean': results['cv_mae_mean'],
            'cv_mae_std': results['cv_mae_std'],
            'n_features': len(final_features)
        }
        
        all_results.append(model_summary)
        best_models.append(results)
        
        # Print results
        print(f"\nResults for {target}:")
        print(f"  Training R²: {results['train_r2']:.4f}")
        print(f"  Validation R²: {results['val_r2']:.4f}")
        print(f"  Test R²: {test_results['test_r2']:.4f}")
        print(f"  CV R²: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        print(f"  Training RMSE: {results['train_rmse']:.4f}")
        print(f"  Validation RMSE: {results['val_rmse']:.4f}")
        print(f"  Test RMSE: {test_results['test_rmse']:.4f}")
        print(f"  CV RMSE: {results['cv_rmse_mean']:.4f} ± {results['cv_rmse_std']:.4f}")
        print(f"  Training MAE: {results['train_mae']:.4f}")
        print(f"  Validation MAE: {results['val_mae']:.4f}")
        print(f"  Test MAE: {test_results['test_mae']:.4f}")
        print(f"  CV MAE: {results['cv_mae_mean']:.4f} ± {results['cv_mae_std']:.4f}")
    
    # Save results
    save_results(all_results, best_models, final_features)
    
    print(f"\n{'='*60}")
    print("MODELING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")

def save_results(all_results, best_models, final_features):
    """Save all results to files."""
    
    # Save model summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUT_DIR, "linear_regression_results.csv"), index=False)
    
    # Save best models summary
    best_models_summary = []
    for i, result in enumerate(all_results):
        best_models_summary.append({
            'target_variable': result['target_variable'],
            'best_validation_r2': result['val_r2'],
            'best_validation_rmse': result['val_rmse'],
            'best_validation_mae': result['val_mae'],
            'cv_r2_mean': result['cv_r2_mean'],
            'cv_r2_std': result['cv_r2_std'],
            'cv_rmse_mean': result['cv_rmse_mean'],
            'cv_rmse_std': result['cv_rmse_std'],
            'cv_mae_mean': result['cv_mae_mean'],
            'cv_mae_std': result['cv_mae_std'],
            'test_r2': result['test_r2'],
            'test_rmse': result['test_rmse'],
            'test_mae': result['test_mae'],
            'n_features': result['n_features']
        })
    
    best_models_df = pd.DataFrame(best_models_summary)
    best_models_df.to_csv(os.path.join(OUT_DIR, "best_models_summary.csv"), index=False)
    
    # Save feature importance for each model
    feature_importance_data = []
    for i, result in enumerate(best_models):
        target = result['model_name']
        importance_df = result['feature_importance'].copy()
        importance_df['target_variable'] = target
        importance_df['rank'] = range(1, len(importance_df) + 1)
        feature_importance_data.append(importance_df)
    
    feature_importance_df = pd.concat(feature_importance_data, ignore_index=True)
    feature_importance_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    
    # Save comprehensive results report
    report_path = os.path.join(OUT_DIR, "linear_regression_results.txt")
    with open(report_path, 'w') as f:
        f.write("LINEAR REGRESSION MODELING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Modeling completed for {len(DEPENDENT_VARS)} dependent variables\n")
        f.write(f"Cross-validation: {CV_FOLDS}-fold\n")
        f.write(f"Final feature count: {len(final_features)}\n")
        f.write(f"Features used: {', '.join(final_features)}\n\n")
        
        for result in all_results:
            f.write(f"Results for {result['target_variable']}:\n")
            f.write(f"  Training R²: {result['train_r2']:.4f}\n")
            f.write(f"  Validation R²: {result['val_r2']:.4f}\n")
            f.write(f"  Test R²: {result['test_r2']:.4f}\n")
            f.write(f"  CV R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}\n")
            f.write(f"  Training RMSE: {result['train_rmse']:.4f}\n")
            f.write(f"  Validation RMSE: {result['val_rmse']:.4f}\n")
            f.write(f"  Test RMSE: {result['test_rmse']:.4f}\n")
            f.write(f"  CV RMSE: {result['cv_rmse_mean']:.4f} ± {result['cv_rmse_std']:.4f}\n")
            f.write(f"  Training MAE: {result['train_mae']:.4f}\n")
            f.write(f"  Validation MAE: {result['val_mae']:.4f}\n")
            f.write(f"  Test MAE: {result['test_mae']:.4f}\n")
            f.write(f"  CV MAE: {result['cv_mae_mean']:.4f} ± {result['cv_mae_std']:.4f}\n")
            f.write(f"  Number of features: {result['n_features']}\n\n")
    
    print(f"\nResults saved to:")
    print(f"  - {OUT_DIR}/linear_regression_results.csv")
    print(f"  - {OUT_DIR}/best_models_summary.csv")
    print(f"  - {OUT_DIR}/feature_importance.csv")
    print(f"  - {OUT_DIR}/linear_regression_results.txt")

if __name__ == "__main__":
    main()
