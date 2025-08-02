"""
Train and evaluate models.
"""

import os
import sys 
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
import optuna

from src.models.linear_regression import LinearRegressionModel
from src.utils.data_processor import DataLoader
from src.utils.config import Settings
from src.utils.logger import setup_logger
import psutil as ps

# Setup logging
logger = setup_logger(__name__, log_file="./logs/models.log")

def train_linear_regression(X_train, y_train, X_val, y_val, trial=None):
    """Train linear regression model with optional hyperparameter optimization."""
    # Hyperparameters
    if trial:
        regularization = trial.suggest_categorical('regularization', ['none', 'ridge', 'lasso'])
        alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True) if regularization != 'none' else 1.0
    else:
        regularization = 'ridge'
        alpha = 1.0
    
    # Create and train model
    model = LinearRegressionModel()
    
    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_param('algorithm ', 'linear_regression')
        mlflow.log_param('regularization ', regularization)
        mlflow.log_param('alpha ', alpha)
        
        # Train model
        train_metrics = model.train(
            X_train, y_train,
            regularization=regularization,
            alpha=alpha
        )
        
        # Evaluate on validation set
        val_metrics = model.evaluate(X_val, y_val)
        
        # Log metrics
        for key, value in train_metrics.items():
            mlflow.log_metric(f'train_{key}' if not key.startswith('train_') else key, value)
        
        for key, value in val_metrics.items():
            mlflow.log_metric(f'val_{key}', value)
        
        # Log model
        mlflow.sklearn.log_model(
            model.model,
            "model",
            registered_model_name="housing_price_predictor"
        )
        
        logger.info(f"Linear Regression - Val R²: {val_metrics['r2']:.4f}, Val RMSE: {val_metrics['rmse']:.2f}")
        
        return val_metrics['r2']  # Return metric for optimization

'''
def train_decision_tree(X_train, y_train, X_val, y_val, trial=None):
    """Train decision tree model with optional hyperparameter optimization."""
    # Hyperparameters
    if trial:
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    else:
        max_depth = 10
        min_samples_split = 5
        min_samples_leaf = 2
    
    # Create and train model
    model = DecisionTreeModel()
    
    with mlflow.start_run(nested=True):
        # Log parameters
        mlflow.log_param('algorithm', 'decision_tree')
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('min_samples_leaf', min_samples_leaf)
        
        # Train model
        train_metrics = model.train(
            X_train, y_train,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # Evaluate on validation set
        val_metrics = model.evaluate(X_val, y_val)
        
        # Log metrics
        for key, value in train_metrics.items():
            mlflow.log_metric(f'train_{key}' if not key.startswith('train_') else key, value)
        
        for key, value in val_metrics.items():
            mlflow.log_metric(f'val_{key}', value)
        
        # Log model
        mlflow.sklearn.log_model(
            model.model,
            "model",
            registered_model_name="housing_price_predictor"
        )
        
        logger.info(f"Decision Tree - Val R²: {val_metrics['r2']:.4f}, Val RMSE: {val_metrics['rmse']:.2f}")
        
        return val_metrics['r2']
'''

'''
def optimize_hyperparameters(algorithm, X_train, y_train, X_val, y_val, n_trials=100):
    """Optimize hyperparameters using Optuna."""
    logger.info(f"Starting hyperparameter optimization for {algorithm}")
    
    def objective(trial):
        if algorithm == 'linear_regression':
            return train_linear_regression(X_train, y_train, X_val, y_val, trial)
        elif algorithm == 'decision_tree':
            return train_decision_tree(X_train, y_train, X_val, y_val, trial)
        elif algorithm == 'random_forest':
            return train_random_forest(X_train, y_train, X_val, y_val, trial)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best parameters for {algorithm}: {study.best_params}")
    logger.info(f"Best score for {algorithm}: {study.best_value:.4f}")
    
    return study.best_params, study.best_value
'''
def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='ML models training for housing price prediction')
    parser.add_argument('--data-path', type=str, help='Path to training dataset')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=5, help='Number of optimization trials')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['linear_regression', 'decision_tree'],
                       default=['linear_regression'],
                       help='Select an algorithms to train the model')
    
    args = parser.parse_args()
    
    # Setup
    settings = Settings()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    
    # Load data
    data_loader = DataLoader()
  
    if args.data_path:
        X, y = data_loader.load_data(args.data_path)   
    
     # Split data
    X_train, X_temp, y_train, y_temp = data_loader.split_data(X,y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = data_loader.split_data(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train models
    results = {}
    
    with mlflow.start_run(run_name="model_comparison"):
        for algorithm in args.algorithms:
            logger.info(f"Starting Training {algorithm}")

            cpu_percent = ps.cpu_percent(interval=1)
            memory_percent = ps.virtual_memory().percent
            disk_percent = ps.disk_usage('/').percent

            mlflow.log_metric("cpu_usage ", cpu_percent)
            mlflow.log_metric("ram_usage ", memory_percent)
            mlflow.log_metric("disk_percent ", disk_percent)

            print(args.optimize)
           
            if args.optimize:
                # Hyperparameter optimization
                best_params, best_score = optimize_hyperparameters(
                    algorithm, X_train, y_train, X_val, y_val, args.n_trials
                )
                results[algorithm] = {'best_params': best_params, 'best_score': best_score}
                
                # Train final model with best parameters
                if algorithm == 'linear_regression':
                    final_score = train_linear_regression(X_train, y_train, X_val, y_val)
                #elif algorithm == 'decision_tree':
                #    final_score = train_decision_tree(X_train, y_train, X_val, y_val)                               
            else:
                # Train with default parameters
                if algorithm == 'linear_regression':
                    score = train_linear_regression(X_train, y_train, X_val, y_val)
                #elif algorithm == 'decision_tree':
                #    score = train_decision_tree(X_train, y_train, X_val, y_val)
                               
                results[algorithm] = {'score': score}
    
    # Print results summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING RESULTS SUMMARY")
    logger.info("="*50)
    
    for algorithm, result in results.items():
        if 'best_score' in result:
            logger.info(f"{algorithm}: Best R² = {result['best_score']:.4f}")
            logger.info(f"  Best params: {result['best_params']}")
        else:
            logger.info(f"{algorithm}: R² = {result['score']:.4f}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()