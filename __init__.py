"""
MLB Game Prediction Package

A modular machine learning system for predicting MLB game outcomes
and evaluating betting strategies.

Modules
-------
data_loader : Data acquisition and preparation
features : Feature engineering and calculation
modeling : Model training and evaluation
betting : Betting strategy simulation
predict : Single game prediction
pipeline : End-to-end workflow orchestration

Classes
-------
MLBPredictor : Main interface for the prediction system

Example Usage
-------------
>>> from mlb_predictor import MLBPredictor
>>> 
>>> predictor = MLBPredictor()
>>> results = predictor.run_complete_pipeline([2018, 2019, 2020])
>>> 
>>> prediction = predictor.predict_game(
...     home_team="NYA",
...     visiting_team="BOS",
...     game_date="20240601",
...     odds={'home_moneyline': -150, 'away_moneyline': +130}
... )
"""

from .predictor import MLBPredictor

__version__ = "1.0.0"
__author__ = "MLB Prediction Team"

__all__ = ['MLBPredictor']
