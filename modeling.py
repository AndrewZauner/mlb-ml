"""
Modeling module for MLB prediction.

This module handles model training, evaluation, and feature importance analysis.

TODO: Experiment with alternative models (XGBoost, LightGBM, neural networks)
TODO: Implement proper hyperparameter optimization (Bayesian optimization)
TODO: Add calibration curves and reliability diagrams
TODO: Implement ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, brier_score_loss,
                            log_loss)
from typing import Tuple, Dict, Optional
import warnings


def prepare_model_data(feature_df: pd.DataFrame,
                       test_size: float = 0.2,
                       min_year: int = 2010) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                       pd.Series, pd.Series]:
    """
    Prepare data for model training by selecting features and splitting.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame with engineered features
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2)
    min_year : int, optional
        Minimum year to include (default: 2010 for recent trends)
    
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
        
    Notes
    -----
    Uses chronological split (not random) to prevent data leakage.
    This is crucial for time series data.
    
    OVERSIMPLIFICATION WARNING:
    - Currently using a simple 80/20 chronological split
    - No separate validation set for hyperparameter tuning
    - Not accounting for seasonal effects (playoff vs regular season)
    
    TODO: Implement walk-forward validation for more robust evaluation
    TODO: Add cross-validation that respects temporal ordering
    TODO: Create separate validation set for hyperparameter tuning
    """
    # Feature selection
    # These are the core features that have proven predictive
    features = [
        # Vegas odds features (typically the strongest signal)
        'home_implied_prob_normalized', 
        'away_implied_prob_normalized',
        
        # Short-term team performance (5 games)
        'home_rolling_5_runs_scored', 
        'home_rolling_5_runs_allowed', 
        'home_rolling_5_win_pct',
        'visiting_rolling_5_runs_scored', 
        'visiting_rolling_5_runs_allowed', 
        'visiting_rolling_5_win_pct',
        
        # Medium-term performance (10 games)
        'home_rolling_10_win_pct', 
        'visiting_rolling_10_win_pct',
        
        # Longer-term performance (20 games)
        'home_rolling_20_win_pct', 
        'visiting_rolling_20_win_pct',
        
        # Comparison features
        'runs_scored_diff_5', 
        'runs_allowed_diff_5', 
        'win_pct_diff_5',
        'runs_scored_diff_10', 
        'runs_allowed_diff_10', 
        'win_pct_diff_10',
        'home_rolling_5_run_diff', 
        'visiting_rolling_5_run_diff',
        
        # Rest and game conditions
        'days_rest_advantage', 
        'is_night_game',
        
        # Day of week (some teams perform better on certain days)
        'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
        
        # Month (captures seasonal trends, weather)
        'month_4', 'month_5', 'month_6', 'month_7', 
        'month_8', 'month_9', 'month_10'
    ]
    
    # Filter to available features
    available_features = [f for f in features if f in feature_df.columns]
    
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: {len(missing)} features not found: {missing}")
    
    # Filter to recent years only (older data may not be relevant)
    # TODO: Experiment with different time windows
    # TODO: Consider weighting recent years more heavily
    recent_data = feature_df[feature_df['season'] >= min_year].copy()
    
    print(f"Using {len(recent_data)} games from {min_year} onwards")
    
    X = recent_data[available_features]
    y = recent_data['home_win']
    
    # CRITICAL: Chronological split for time series data
    # We use the first 80% for training and last 20% for testing
    # This simulates real-world usage where we predict future games
    train_cutoff = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:train_cutoff], X.iloc[train_cutoff:]
    y_train, y_test = y.iloc[:train_cutoff], y.iloc[train_cutoff:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Class balance in training: {y_train.mean():.3f} (home win rate)")
    print(f"Class balance in testing: {y_test.mean():.3f} (home win rate)")
    
    # TODO: Check for class imbalance and consider SMOTE or class weights
    # Baseball typically has ~54% home win rate, so slight imbalance exists
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, 
                y_train: pd.Series,
                grid_search: bool = True,
                random_state: int = 42) -> Pipeline:
    """
    Train a machine learning model for game prediction.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    grid_search : bool, optional
        Whether to perform grid search for hyperparameter tuning
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Trained model pipeline
        
    Notes
    -----
    Uses GradientBoostingClassifier by default as it typically
    performs well on structured data with mixed feature types.
    
    CURRENT MODEL CHOICE:
    - GradientBoostingClassifier: Good baseline, interpretable
    - StandardScaler: Normalizes features (important for some models)
    
    IMPROVEMENTS IMPLEMENTED:
    - Added random_state for reproducibility
    - Included StandardScaler in pipeline
    - Grid search over key hyperparameters
    
    TODO: Model variants to try:
    1. XGBoost or LightGBM (often outperform sklearn GBM)
       - Better handling of missing values
       - Built-in regularization
       - Faster training
       
    2. Neural networks (MLPClassifier or deep learning)
       - Can capture complex nonlinear interactions
       - May need more data to avoid overfitting
       
    3. Ensemble methods:
       - Stack multiple models (GBM + RF + Logistic)
       - Voting classifier
       - Weighted averaging based on recent performance
       
    4. Calibration:
       - CalibratedClassifierCV to improve probability estimates
       - Important for betting applications
       
    TODO: Hyperparameter optimization approaches:
    - Bayesian optimization (scikit-optimize, Optuna)
    - Randomized search for faster initial exploration
    - Nested cross-validation for unbiased performance estimates
    
    TODO: Handle class imbalance:
    - Set class_weight='balanced' 
    - Or use SMOTE for synthetic oversampling
    """
    print("Training model...")
    
    if grid_search:
        # Define pipeline with scaling and classification
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=random_state))
        ])
        
        # Hyperparameter grid
        # IMPROVEMENT: Expanded from original with better ranges
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],  # More trees generally better
            'classifier__learning_rate': [0.01, 0.05, 0.1],  # Smaller = more conservative
            'classifier__max_depth': [3, 4, 5],  # Depth controls complexity
            'classifier__min_samples_split': [2, 5],  # Regularization parameter
            'classifier__subsample': [0.8, 1.0],  # Stochastic gradient boosting
        }
        
        # TODO: Use RandomizedSearchCV for faster search with more parameters
        # TODO: Implement Bayesian optimization for more efficient search
        
        # Perform grid search with cross-validation
        # Note: cv=5 may not respect temporal ordering - consider TimeSeriesSplit
        # TODO: Replace with TimeSeriesSplit for proper time series CV
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5,  # TODO: Use TimeSeriesSplit instead
            scoring='neg_log_loss',  # Better for probability calibration than accuracy
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best CV score: {-grid.best_score_:.4f} (log loss)")
        
    else:
        # Use fixed hyperparameters (faster, good for initial testing)
        # IMPROVEMENT: Better default parameters than before
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,  # Slightly lower for stability
                max_depth=4,  # Moderate depth
                min_samples_split=5,  # Prevent overfitting
                subsample=0.8,  # Stochastic boosting
                random_state=random_state
            ))
        ])
        
        model = pipeline.fit(X_train, y_train)
    
    print("Model training complete.")
    return model


def evaluate_model(model: Pipeline,
                   X_test: pd.DataFrame, 
                   y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics
        
    Notes
    -----
    IMPROVEMENT: Added additional metrics beyond the original:
    - Brier score: Measures quality of probabilistic predictions
    - Log loss: Penalizes confident wrong predictions
    
    TODO: Add calibration analysis
    - Plot calibration curves
    - Compute Expected Calibration Error (ECE)
    
    TODO: Analyze performance by subgroups:
    - Home favorites vs underdogs
    - High vs low scoring games
    - Different months/weather conditions
    - Team strength tiers
    """
    if model is None:
        print("Model not trained yet.")
        return None
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),  # NEW: probability calibration
        'log_loss': log_loss(y_test, y_pred_proba),  # NEW: better for betting
    }
    
    print("\nModel Evaluation Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:.<30} {value:.4f}")
    print("=" * 50)
    
    # Baseline comparison
    # In baseball, home teams win ~54% of games
    baseline_accuracy = y_test.mean() if y_test.mean() > 0.5 else 1 - y_test.mean()
    print(f"\nBaseline (always predict home/away): {baseline_accuracy:.4f}")
    print(f"Model improvement: {(metrics['accuracy'] - baseline_accuracy):.4f}")
    
    # TODO: Add confusion matrix analysis
    # TODO: Plot ROC curve and precision-recall curve
    # TODO: Analyze predictions by confidence level
    
    return metrics


def feature_importance(model: Pipeline) -> pd.DataFrame:
    """
    Extract and display feature importances from the trained model.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline
    
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores
        
    Notes
    -----
    Feature importance helps identify which factors are most predictive.
    
    Expected top features (based on domain knowledge):
    1. Vegas implied probabilities (strongest signal)
    2. Recent win percentage
    3. Run differential
    4. Rest days advantage
    
    TODO: Implement SHAP values for better feature importance
    - More accurate attribution than built-in importances
    - Shows direction of effect (positive/negative)
    - Can explain individual predictions
    
    TODO: Analyze feature interactions
    - Which features work together?
    - Are there redundant features?
    """
    if model is None:
        print("Model not trained yet.")
        return None
    
    try:
        # Get feature names and importances from the classifier step
        feature_names = model.named_steps['classifier'].feature_names_in_
        importances = model.named_steps['classifier'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        print("\nTop 15 Most Important Features:")
        print("=" * 60)
        for idx, row in importance_df.head(15).iterrows():
            bar = '█' * int(row['importance'] * 200)  # Visual bar
            print(f"{row['feature']:.<40} {row['importance']:.4f} {bar}")
        print("=" * 60)
        
        return importance_df
        
    except Exception as e:
        print(f"Error extracting feature importances: {e}")
        return None
