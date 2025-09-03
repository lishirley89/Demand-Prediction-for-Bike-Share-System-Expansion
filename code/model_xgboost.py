#!/usr/bin/env python3
"""
XGBoost Model for Bike Station Demand Prediction

This script builds separate XGBoost models for different dependent variables:
- cbike_start
- cbike_end  
- ebike_start
- ebike_end

Features:
- Grid search hyperparameter tuning
- 5-fold cross-validation
- 80/20 train/validation split
- Comprehensive model evaluation
- Best model application to test dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Data is already standardized from preprocessing step
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load preprocessed training and test datasets"""
    print("Loading preprocessed datasets...")
    
    train_data = pd.read_csv('result/training_dataset_preprocessed.csv')
    test_data = pd.read_csv('result/test_dataset_preprocessed.csv')
    
    print(f"Training dataset shape: {train_data.shape}")
    print(f"Test dataset shape: {test_data.shape}")
    
    return train_data, test_data

def prepare_features(train_data, test_data):
    """Prepare features by removing unwanted columns"""
    # Columns to exclude
    exclude_cols = ['station_id', 'year', 'total_start', 'total_end']
    
    # Remove columns that don't exist in the datasets
    exclude_cols = [col for col in exclude_cols if col in train_data.columns]
    
    # Define dependent variables
    target_vars = ['cbike_start', 'cbike_end', 'ebike_start', 'ebike_end']
    
    # Get feature columns (all columns except targets and excluded)
    feature_cols = [col for col in train_data.columns 
                   if col not in target_vars and col not in exclude_cols]
    
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target variables: {target_vars}")
    
    return feature_cols, target_vars

def setup_gpu():
    """Setup and verify GPU availability"""
    import xgboost as xgb
    
    print("Checking GPU availability...")
    
    # Check if GPU is available
    try:
        # Try to create a simple model with GPU
        test_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
        print("✅ GPU detected and available for XGBoost")
        return True
    except Exception as e:
        print(f"❌ GPU not available: {e}")
        print("Falling back to CPU mode")
        return False

def define_parameter_grid(use_gpu=True):
    """Define parameter grid for XGBoost hyperparameter tuning (GPU optimized)"""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    
    if use_gpu:
        param_grid.update({
            'tree_method': ['gpu_hist'],
            'gpu_id': [0]
        })
    
    return param_grid

def train_xgboost_model(X_train, y_train, param_grid, cv_folds=5, use_gpu=True):
    """Train XGBoost model with grid search and cross-validation (GPU optimized)"""
    print("Training XGBoost model with grid search...")
    
    # Initialize XGBoost regressor with GPU support
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=1,  # Reduce to 1 for GPU (parallelization handled by GPU)
        early_stopping_rounds=50
    )
    
    # Add GPU parameters if available
    if use_gpu:
        xgb_model.set_params(tree_method='gpu_hist', gpu_id=0)
        print("🚀 Using GPU acceleration")
    else:
        print("💻 Using CPU mode")
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=1,  # Reduce to 1 for GPU
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target_name):
    """Evaluate model performance on train, validation, and test sets"""
    print(f"\nEvaluating {target_name} model...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()
    
    cv_rmse_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_mean = np.sqrt(-cv_rmse_scores.mean())
    cv_rmse_std = np.sqrt(cv_rmse_scores.var())
    
    cv_mae_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae_mean = -cv_mae_scores.mean()
    cv_mae_std = cv_mae_scores.std()
    
    # Print results
    print(f"Training R²: {train_r2:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"CV R²: {cv_r2_mean:.4f} (±{cv_r2_std:.4f})")
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"CV RMSE: {cv_rmse_mean:.4f} (±{cv_rmse_std:.4f})")
    
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"CV MAE: {cv_mae_mean:.4f} (±{cv_mae_std:.4f})")
    
    # Return results dictionary
    results = {
        'target_variable': target_name,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'test_rmse': test_rmse,
        'cv_rmse_mean': cv_rmse_mean,
        'cv_rmse_std': cv_rmse_std,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'cv_mae_mean': cv_mae_mean,
        'cv_mae_std': cv_mae_std
    }
    
    return results, model

def get_feature_importance(model, feature_names):
    """Extract feature importance from the trained model"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df

def main():
    """Main execution function"""
    print("=" * 60)
    print("XGBoost Model for Bike Station Demand Prediction (GPU Optimized)")
    print("=" * 60)
    
    # Load data
    train_data, test_data = load_data()
    
    # Prepare features
    feature_cols, target_vars = prepare_features(train_data, test_data)
    
    # Prepare X and y for training
    X_train_full = train_data[feature_cols]
    y_train_full = train_data[target_vars]
    
    # Prepare test data
    X_test = test_data[feature_cols]
    y_test = test_data[target_vars]
    
    # Data is already standardized from preprocessing step
    X_train_scaled = X_train_full
    X_test_scaled = X_test
    
    # Setup GPU
    use_gpu = setup_gpu()
    
    # Define parameter grid (GPU-aware)
    param_grid = define_parameter_grid(use_gpu)
    
    # Store results
    all_results = []
    all_feature_importance = []
    best_models = {}
    
    # Train models for each target variable
    for target in target_vars:
        print(f"\n{'='*50}")
        print(f"Training model for: {target}")
        print(f"{'='*50}")
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_scaled, y_train_full[target], 
            test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        
        # Train model
        best_model, best_params = train_xgboost_model(X_train, y_train, param_grid, use_gpu=use_gpu)
        
        # Evaluate model
        results, trained_model = evaluate_model(
            best_model, X_train, y_train, X_val, y_val, 
            X_test_scaled, y_test[target], target
        )
        
        # Store results
        all_results.append(results)
        best_models[target] = trained_model
        
        # Get feature importance
        feature_importance = get_feature_importance(trained_model, feature_cols)
        feature_importance['target_variable'] = target
        all_feature_importance.append(feature_importance)
        
        print(f"\nTop 10 features for {target}:")
        print(feature_importance.head(10))
    
    # Save results
    print("\nSaving results...")
    
    # Save model performance results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('result/xgboost_model_results.csv', index=False)
    print("Model results saved to: result/xgboost_model_results.csv")
    
    # Save feature importance
    feature_importance_df = pd.concat(all_feature_importance, ignore_index=True)
    feature_importance_df.to_csv('result/xgboost_feature_importance.csv', index=False)
    print("Feature importance saved to: result/xgboost_feature_importance.csv")
    
    # Save detailed results report
    with open('result/xgboost_results_report.txt', 'w') as f:
        f.write("XGBoost Model Results Report\n")
        f.write("=" * 50 + "\n\n")
        
        for result in all_results:
            f.write(f"Target Variable: {result['target_variable']}\n")
            f.write(f"Training R²: {result['train_r2']:.4f}\n")
            f.write(f"Validation R²: {result['val_r2']:.4f}\n")
            f.write(f"Test R²: {result['test_r2']:.4f}\n")
            f.write(f"CV R²: {result['cv_r2_mean']:.4f} (±{result['cv_r2_std']:.4f})\n")
            f.write(f"Training RMSE: {result['train_rmse']:.4f}\n")
            f.write(f"Validation RMSE: {result['val_rmse']:.4f}\n")
            f.write(f"Test RMSE: {result['test_rmse']:.4f}\n")
            f.write(f"CV RMSE: {result['cv_rmse_mean']:.4f} (±{result['cv_rmse_std']:.4f})\n")
            f.write(f"Training MAE: {result['train_mae']:.4f}\n")
            f.write(f"Validation MAE: {result['val_mae']:.4f}\n")
            f.write(f"Test MAE: {result['test_mae']:.4f}\n")
            f.write(f"CV MAE: {result['cv_mae_mean']:.4f} (±{result['cv_mae_std']:.4f})\n")
            f.write("\n" + "-" * 30 + "\n\n")
    
    print("Detailed results report saved to: result/xgboost_results_report.txt")
    
    # Print summary
    print("\n" + "=" * 60)
    print("XGBoost Modeling Complete!")
    print("=" * 60)
    
    print("\nModel Performance Summary:")
    for result in all_results:
        print(f"{result['target_variable']}:")
        print(f"  Test R²: {result['test_r2']:.4f}")
        print(f"  Test RMSE: {result['test_rmse']:.4f}")
        print(f"  Test MAE: {result['test_mae']:.4f}")
        print()

if __name__ == "__main__":
    main()
