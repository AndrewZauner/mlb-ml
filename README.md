# MLB Game Prediction Model

A machine learning system for predicting MLB game outcomes and evaluating betting strategies.

## Project Structure

```
mlb_predictor/
├── __init__.py           # Package initialization
├── predictor.py          # Main MLBPredictor class
├── data_loader.py        # Data acquisition and loading
├── features.py           # Feature engineering
├── modeling.py           # Model training and evaluation
├── betting.py            # Betting strategy simulation
├── predict.py            # Single game prediction
└── pipeline.py           # End-to-end workflow orchestration

main.py                   # Entry point for running the pipeline
data/                     # Downloaded Retrosheet data (created automatically)
```

## Installation

### Requirements

```bash
pip install pandas numpy scikit-learn requests
```

### Optional (for enhancements)

```bash
pip install xgboost lightgbm shap joblib
```

## Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Download historical game data from Retrosheet
2. Generate placeholder odds data
3. Engineer features
4. Train a gradient boosting model
5. Evaluate model performance
6. Simulate betting strategies
7. Demonstrate a single game prediction

## Usage Examples

### Basic Usage

```python
from mlb_predictor import MLBPredictor

# Initialize
predictor = MLBPredictor(data_dir='./data')

# Run complete pipeline
results = predictor.run_complete_pipeline([2018, 2019, 2020, 2021, 2022])

# Make a prediction
prediction = predictor.predict_game(
    home_team="NYA",
    visiting_team="BOS",
    game_date="20240601",
    odds={'home_moneyline': -150, 'away_moneyline': +130}
)

print(f"Home win probability: {prediction['home_win_probability']:.1%}")
print(f"Recommendation: {prediction['recommendation']}")
```

### Step-by-Step Usage

```python
from mlb_predictor import MLBPredictor

predictor = MLBPredictor()

# 1. Download and load data
predictor.download_retrosheet_data([2020, 2021, 2022])
game_data = predictor.load_retrosheet_data([2020, 2021, 2022])

# 2. Get odds and merge
odds_data = predictor.download_odds_data([2020, 2021, 2022])
merged = predictor.merge_game_and_odds_data()

# 3. Engineer features
features = predictor.engineer_features(merged)

# 4. Prepare data and train
X_train, X_test, y_train, y_test = predictor.prepare_model_data(features)
model = predictor.train_model(X_train, y_train)

# 5. Evaluate
metrics = predictor.evaluate_model(X_test, y_test)
importance = predictor.feature_importance()
```

## Key Features

### Data Loading (`data_loader.py`)
- Downloads game logs from Retrosheet
- Properly maps batting statistics columns (fixes KeyError bug)
- Merges game data with odds data
- **BUG FIX**: Now correctly extracts `home_H`, `home_AB`, etc. from Retrosheet files

### Feature Engineering (`features.py`)
- Rolling team statistics (5, 10, 20 game windows)
- Implied probabilities from betting odds
- Temporal features (day of week, month, rest days)
- Batting statistics and run differentials
- **DEFENSIVE**: Checks for missing columns before use

### Modeling (`modeling.py`)
- Gradient Boosting Classifier with grid search
- Chronological train/test split (prevents data leakage)
- Multiple evaluation metrics (accuracy, AUC, Brier score, log loss)
- Feature importance analysis

### Betting Simulation (`betting.py`)
- Kelly Criterion bet sizing
- Minimum edge thresholds
- Bankroll management
- Performance by confidence level
- **NOTE**: Currently assumes fair odds without vig

### Prediction (`predict.py`)
- Single game outcome prediction
- Edge calculation vs. market odds
- Betting recommendations

## What Was Fixed

### 1. KeyError: 'home_H' Bug
**Problem**: The original code assumed batting stat columns existed but Retrosheet loading didn't create them.

**Solution**: 
- Added proper column mapping based on Retrosheet field specification
- Maps columns 25-29 (visiting team stats) and 48-52 (home team stats)
- Defensive checks in feature engineering
- Graceful fallbacks when columns are missing

### 2. Code Organization
**Before**: Single 950-line file

**After**: Modular structure with 7 focused modules
- Better separation of concerns
- Easier to test and maintain
- Clear extension points

## Areas for Improvement

The code includes extensive TODO comments marking oversimplifications. Key areas:

### High Priority
1. **Replace placeholder odds** with real historical data
   - Current: Random odds generation
   - Needed: Real sportsbook data via API or scraping

2. **Add pitcher statistics**
   - Current: Team-level stats only
   - Needed: Starting pitcher ERA, WHIP, K/9, recent performance

3. **Implement park factors**
   - Current: All ballparks treated equally
   - Needed: Park-adjusted offensive/pitching metrics

### Medium Priority
4. **Better hyperparameter tuning**
   - Consider Bayesian optimization (Optuna)
   - Try XGBoost or LightGBM

5. **Improve bet sizing**
   - Current: Simplified Kelly with fair odds
   - Needed: Kelly with actual vig-adjusted odds

6. **Add recency weighting**
   - Current: Simple moving averages
   - Needed: Exponentially weighted stats (recent games matter more)

### Lower Priority (But Still Important)
7. Walk-forward validation
8. Calibration curves
9. SHAP values for feature importance
10. Injury/roster data
11. Weather features
12. Multiple model ensemble

## Extending the Model

Each module has clear extension points marked with TODO comments:

### Adding a New Feature
Edit `mlb_predictor/features.py`:

```python
def engineer_features(data, window_sizes=[5, 10, 20]):
    # ... existing code ...
    
    # Add your new feature here
    df['new_feature'] = calculate_new_feature(df)
    
    return feature_df
```

Then add it to the feature list in `mlb_predictor/modeling.py`:

```python
features = [
    'home_implied_prob_normalized',
    # ... existing features ...
    'new_feature',  # Add here
]
```

### Trying a Different Model
Edit `mlb_predictor/modeling.py`:

```python
def train_model(X_train, y_train, grid_search=True):
    # Replace GradientBoostingClassifier with your model
    from xgboost import XGBClassifier
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier())
    ])
    # ... rest of function
```

### Adding Real Odds Data
Edit `mlb_predictor/data_loader.py`:

```python
def download_odds_data(years):
    # Replace placeholder with API call
    import requests
    
    odds_data = []
    for year in years:
        response = requests.get(f"https://api.oddsapi.com/v1/{year}")
        # Parse response...
        odds_data.append(parsed_data)
    
    return pd.concat(odds_data)
```

## Performance Expectations

### Model Accuracy
- Baseline (always predict home): ~54%
- Expected model accuracy: 55-57%
- **Note**: Even small improvements over baseline are valuable for betting

### Betting Performance
- With perfect predictions at 57% accuracy and 3% edge requirement
- Expected ROI: 5-15% (but highly variable)
- **WARNING**: Current results use placeholder odds and may not reflect real performance

### Important Caveats
1. Past performance doesn't guarantee future results
2. Sportsbooks adjust lines based on sharp money
3. Vig reduces theoretical edge by 4-5%
4. Line shopping and timing matter significantly
5. Always bet responsibly with money you can afford to lose

## Contributing

When adding features or making changes:

1. **Document oversimplifications** - Add TODO comments explaining what could be better
2. **Maintain modularity** - Keep functions focused and modules independent
3. **Add docstrings** - Explain parameters, returns, and any gotchas
4. **Preserve the external API** - MLBPredictor class should remain stable
5. **Test chronological splits** - Never shuffle time series data

## License

This is an educational project. Use at your own risk. Sports betting carries risk of loss.

## Acknowledgments

- Game data from [Retrosheet](https://www.retrosheet.org/)
- Inspired by the sports analytics community
- Built with scikit-learn, pandas, and numpy
