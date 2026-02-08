import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import requests
import os
import zipfile
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLBPredictor:
    def __init__(self, data_dir='./data'):
        """
        Initialize the MLB game prediction model
        
        Parameters:
        -----------
        data_dir : str
            Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.game_data = None
        self.odds_data = None
        self.model = None
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def download_retrosheet_data(self, years):
        """
        Download game data from Retrosheet for specified years
        
        Parameters:
        -----------
        years : list
            List of years to download data for (e.g., [2018, 2019, 2020])
        """
        print(f"Downloading Retrosheet data for years: {years}")
        
        for year in years:
            url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(response.content))
                    z.extractall(self.data_dir)
                    print(f"Successfully downloaded and extracted data for {year}")
                else:
                    print(f"Failed to download data for {year}: Status code {response.status_code}")
            except Exception as e:
                print(f"Error downloading data for {year}: {e}")
    
    def load_retrosheet_data(self, years):
        """
        Load and parse Retrosheet game logs
        
        Parameters:
        -----------
        years : list
            List of years to load data for
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing game data
        """
        all_data = []
        
        for year in years:
            file_path = os.path.join(self.data_dir, f"GL{year}.TXT")
            if os.path.exists(file_path):
                try:
                    # Load data without predefined column names - let pandas detect the number of columns
                    year_data = pd.read_csv(file_path, header=None, sep=',', quotechar='"')
                    
                    # Get the number of columns from the data itself
                    num_columns = year_data.shape[1]
                    
                    # Use generic column names based on the actual number of columns
                    column_names = [f'col_{i}' for i in range(num_columns)]
                    year_data.columns = column_names
                    
                    # Rename essential columns we know the positions of
                    rename_map = {
                        'col_0': 'date',
                        'col_3': 'visiting_team',
                        'col_6': 'home_team',
                        'col_9': 'visiting_score',
                        'col_10': 'home_score',
                        'col_12': 'day_night'
                    }
                    year_data.rename(columns=rename_map, inplace=True)
                    
                    # Add year column
                    year_data['season'] = year
                    
                    # Append to full dataset
                    all_data.append(year_data)
                    print(f"Loaded {len(year_data)} games from {year}")
                except Exception as e:
                    print(f"Error loading data for {year}: {e}")
            else:
                print(f"File not found: {file_path}")
        
        if all_data:
            self.game_data = pd.concat(all_data, ignore_index=True)
            print(f"Total games loaded: {len(self.game_data)}")
            return self.game_data
        else:
            print("No data was loaded.")
            return None
    
    def download_odds_data(self, years):
        """
        Placeholder for downloading historical odds data
        In practice, you would need to scrape or use an API for historical odds
        
        Parameters:
        -----------
        years : list
            List of years to download odds data for
        """
        print("Note: This is a placeholder function for downloading odds data.")
        print("In a real implementation, you would need to:")
        print("1. Subscribe to a sports odds API (e.g., OddsAPI, The Odds API)")
        print("2. Scrape historical odds from a site like OddsShark or Vegas Insider")
        print("3. Purchase historical odds data from a provider")
        
        # Create a sample odds dataframe with placeholder data
        # In reality, you would load actual odds data here
        dates = []
        home_teams = []
        away_teams = []
        home_moneyline = []
        away_moneyline = []
        
        # Generate some placeholder data
        # This should be replaced with actual data in production
        for year in years:
            for month in range(4, 11):  # Baseball season (April-October)
                for day in range(1, 28):
                    if np.random.random() < 0.3:  # Not every day has games
                        continue
                    
                    # Generate 8 random games for this date
                    for _ in range(8):
                        date_str = f"{year}{month:02d}{day:02d}"
                        teams = np.random.choice(['NYA', 'BOS', 'TOR', 'BAL', 'TBA',
                                                  'CHA', 'CLE', 'DET', 'KCA', 'MIN',
                                                  'HOU', 'LAA', 'OAK', 'SEA', 'TEX',
                                                  'ATL', 'MIA', 'NYN', 'PHI', 'WAS',
                                                  'CHN', 'CIN', 'MIL', 'PIT', 'SLN',
                                                  'ARI', 'COL', 'LAN', 'SDN', 'SFN'], 2, replace=False)
                        home_team = teams[0]
                        away_team = teams[1]
                        
                        # Generate random odds (favorite usually between -110 and -220, underdog between +100 and +200)
                        is_home_favorite = np.random.random() > 0.4  # Home teams are slightly more likely to be favored
                        
                        if is_home_favorite:
                            home_line = -np.random.randint(110, 220)
                            away_line = np.random.randint(100, 200)
                        else:
                            home_line = np.random.randint(100, 200)
                            away_line = -np.random.randint(110, 220)
                        
                        dates.append(date_str)
                        home_teams.append(home_team)
                        away_teams.append(away_team)
                        home_moneyline.append(home_line)
                        away_moneyline.append(away_line)
        
        self.odds_data = pd.DataFrame({
            'date': dates,
            'home_team': home_teams,
            'visiting_team': away_teams,
            'home_moneyline': home_moneyline,
            'away_moneyline': away_moneyline
        })
        
        print(f"Created placeholder odds data with {len(self.odds_data)} entries")
        return self.odds_data
    
    def merge_game_and_odds_data(self):
        """
        Merge game data with odds data
        
        Returns:
        --------
        pandas.DataFrame
            Merged dataframe with game and odds data
        """
        if self.game_data is None or self.odds_data is None:
            print("Game data or odds data not loaded.")
            return None
        
        # Convert date formats to match
        self.game_data['date_str'] = self.game_data['date'].astype(str)
        
        # Merge on date and team names
        merged_data = pd.merge(
            self.game_data,
            self.odds_data,
            left_on=['date_str', 'home_team', 'visiting_team'],
            right_on=['date', 'home_team', 'visiting_team'],
            how='inner'
        )
        
        print(f"Merged data has {len(merged_data)} rows")
        return merged_data
    
    def engineer_features(self, data, window_sizes=[5, 10, 20]):
        """
        Engineer features for the prediction model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Merged game and odds data
        window_sizes : list
            List of window sizes for rolling statistics
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        print("Engineering features...")
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Convert date to datetime
        df['game_date'] = pd.to_datetime(df['date_str'], format='%Y%m%d')
        
        # Create target variable: 1 if home team wins, 0 otherwise
        df['home_win'] = (df['home_score'] > df['visiting_score']).astype(int)
        
        # Basic features: run differential, batting stats
        df['home_run_diff'] = df['home_score'] - df['visiting_score']
        df['home_batting_avg'] = df['home_H'] / df['home_AB']
        df['visiting_batting_avg'] = df['visiting_H'] / df['visiting_AB']
        
        # Handle division by zero
        df['home_batting_avg'] = df['home_batting_avg'].fillna(0)
        df['visiting_batting_avg'] = df['visiting_batting_avg'].fillna(0)
        
        # Create odds-based features
        df['home_implied_prob'] = np.where(
            df['home_moneyline'] > 0,
            100 / (df['home_moneyline'] + 100),
            abs(df['home_moneyline']) / (abs(df['home_moneyline']) + 100)
        )
        
        df['away_implied_prob'] = np.where(
            df['away_moneyline'] > 0,
            100 / (df['away_moneyline'] + 100),
            abs(df['away_moneyline']) / (abs(df['away_moneyline']) + 100)
        )
        
        # Normalize implied probabilities to sum to 1 (accounting for the vig)
        df['total_implied_prob'] = df['home_implied_prob'] + df['away_implied_prob']
        df['home_implied_prob_normalized'] = df['home_implied_prob'] / df['total_implied_prob']
        df['away_implied_prob_normalized'] = df['away_implied_prob'] / df['total_implied_prob']
        
        # Create day/night game indicator
        df['is_night_game'] = (df['day_night'] == 'N').astype(int)
        
        # Sort by date for time-series features
        df = df.sort_values(by=['game_date'])
        
        # Create team-specific DataFrames for rolling stats
        team_groups = {}
        for team in set(df['home_team'].unique()) | set(df['visiting_team'].unique()):
            # Get games where team is home
            home_games = df[df['home_team'] == team].copy()
            home_games['team_score'] = home_games['home_score']
            home_games['opponent_score'] = home_games['visiting_score']
            home_games['is_home'] = 1
            
            # Get games where team is away
            away_games = df[df['visiting_team'] == team].copy()
            away_games['team_score'] = away_games['visiting_score']
            away_games['opponent_score'] = away_games['home_score']
            away_games['is_home'] = 0
            
            # Combine and sort by date
            team_games = pd.concat([home_games, away_games])
            team_games = team_games.sort_values(by='game_date')
            
            # Add to dictionary
            team_groups[team] = team_games
        
        # Calculate rolling stats for each team
        for team, team_df in team_groups.items():
            for window in window_sizes:
                # Calculate rolling stats
                team_df[f'rolling_{window}_runs_scored'] = team_df['team_score'].rolling(window=window, min_periods=1).mean()
                team_df[f'rolling_{window}_runs_allowed'] = team_df['opponent_score'].rolling(window=window, min_periods=1).mean()
                team_df[f'rolling_{window}_win_pct'] = (team_df['team_score'] > team_df['opponent_score']).rolling(window=window, min_periods=1).mean()
            
            # Update the original team groups
            team_groups[team] = team_df
        
        # Create a list to hold the rows for our final DataFrame
        feature_rows = []
        
        # For each game in the original dataset
        for _, game in df.iterrows():
            home_team = game['home_team']
            visiting_team = game['visiting_team']
            game_date = game['game_date']
            
            # Get the home team's stats before this game
            home_team_previous_games = team_groups[home_team][team_groups[home_team]['game_date'] < game_date]
            
            # Get the visiting team's stats before this game
            visiting_team_previous_games = team_groups[visiting_team][team_groups[visiting_team]['game_date'] < game_date]
            
            # If we have previous games for both teams
            if not home_team_previous_games.empty and not visiting_team_previous_games.empty:
                # Get the most recent stats for each team
                home_team_stats = home_team_previous_games.iloc[-1].copy()
                visiting_team_stats = visiting_team_previous_games.iloc[-1].copy()
                
                # Create a new row with the original game data and the added stats
                new_row = game.copy()
                
                # Add the rolling stats for each window size
                for window in window_sizes:
                    # Home team stats
                    new_row[f'home_rolling_{window}_runs_scored'] = home_team_stats[f'rolling_{window}_runs_scored']
                    new_row[f'home_rolling_{window}_runs_allowed'] = home_team_stats[f'rolling_{window}_runs_allowed']
                    new_row[f'home_rolling_{window}_win_pct'] = home_team_stats[f'rolling_{window}_win_pct']
                    
                    # Visiting team stats
                    new_row[f'visiting_rolling_{window}_runs_scored'] = visiting_team_stats[f'rolling_{window}_runs_scored']
                    new_row[f'visiting_rolling_{window}_runs_allowed'] = visiting_team_stats[f'rolling_{window}_runs_allowed']
                    new_row[f'visiting_rolling_{window}_win_pct'] = visiting_team_stats[f'rolling_{window}_win_pct']
                
                # Add to our list of feature rows
                feature_rows.append(new_row)
        
        # Create the final feature DataFrame
        feature_df = pd.DataFrame(feature_rows)
        
        # Create additional derived features
        for window in window_sizes:
            # Run differential features
            feature_df[f'home_rolling_{window}_run_diff'] = feature_df[f'home_rolling_{window}_runs_scored'] - feature_df[f'home_rolling_{window}_runs_allowed']
            feature_df[f'visiting_rolling_{window}_run_diff'] = feature_df[f'visiting_rolling_{window}_runs_scored'] - feature_df[f'visiting_rolling_{window}_runs_allowed']
            
            # Team comparison features
            feature_df[f'runs_scored_diff_{window}'] = feature_df[f'home_rolling_{window}_runs_scored'] - feature_df[f'visiting_rolling_{window}_runs_scored']
            feature_df[f'runs_allowed_diff_{window}'] = feature_df[f'home_rolling_{window}_runs_allowed'] - feature_df[f'visiting_rolling_{window}_runs_allowed']
            feature_df[f'win_pct_diff_{window}'] = feature_df[f'home_rolling_{window}_win_pct'] - feature_df[f'visiting_rolling_{window}_win_pct']
        
        # Add day of week features (one-hot encoded)
        feature_df['day_of_week_num'] = pd.to_datetime(feature_df['date_str']).dt.dayofweek
        day_of_week_dummies = pd.get_dummies(feature_df['day_of_week_num'], prefix='day')
        feature_df = pd.concat([feature_df, day_of_week_dummies], axis=1)
        
        # Add month features (one-hot encoded)
        feature_df['month'] = pd.to_datetime(feature_df['date_str']).dt.month
        month_dummies = pd.get_dummies(feature_df['month'], prefix='month')
        feature_df = pd.concat([feature_df, month_dummies], axis=1)
        
        # Check for rest days (this is a simplified approach - in reality would need more detailed schedule data)
        feature_df = feature_df.sort_values(['home_team', 'game_date'])
        feature_df['home_days_rest'] = feature_df.groupby('home_team')['game_date'].diff().dt.days
        
        feature_df = feature_df.sort_values(['visiting_team', 'game_date'])
        feature_df['visiting_days_rest'] = feature_df.groupby('visiting_team')['game_date'].diff().dt.days
        
        # Replace NaN with median values
        feature_df['home_days_rest'] = feature_df['home_days_rest'].fillna(feature_df['home_days_rest'].median())
        feature_df['visiting_days_rest'] = feature_df['visiting_days_rest'].fillna(feature_df['visiting_days_rest'].median())
        
        # Days rest advantage
        feature_df['days_rest_advantage'] = feature_df['home_days_rest'] - feature_df['visiting_days_rest']
        
        # Clean up any remaining NaN values
        feature_df = feature_df.fillna(0)
        
        print(f"Feature engineering complete. Dataset has {len(feature_df)} rows and {len(feature_df.columns)} columns.")
        return feature_df
    
    def prepare_model_data(self, feature_df):
        """
        Prepare data for model training by selecting relevant features and splitting the data
        
        Parameters:
        -----------
        feature_df : pandas.DataFrame
            DataFrame with engineered features
        
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        # Select features for model training
        features = [
            # Vegas odds features
            'home_implied_prob_normalized', 'away_implied_prob_normalized',
            
            # Team performance features (for each window size)
            'home_rolling_5_runs_scored', 'home_rolling_5_runs_allowed', 'home_rolling_5_win_pct',
            'visiting_rolling_5_runs_scored', 'visiting_rolling_5_runs_allowed', 'visiting_rolling_5_win_pct',
            'home_rolling_10_win_pct', 'visiting_rolling_10_win_pct',
            'home_rolling_20_win_pct', 'visiting_rolling_20_win_pct',
            
            # Derived comparison features
            'runs_scored_diff_5', 'runs_allowed_diff_5', 'win_pct_diff_5',
            'runs_scored_diff_10', 'runs_allowed_diff_10', 'win_pct_diff_10',
            'home_rolling_5_run_diff', 'visiting_rolling_5_run_diff',
            
            # Rest and location features
            'days_rest_advantage', 'is_night_game',
            
            # Day of week features
            'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
            
            # Month features
            'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10'
        ]
        
        # Ensure all features exist in the DataFrame
        available_features = [f for f in features if f in feature_df.columns]
        
        # Select only games from 2010 onwards for more recent trends
        recent_data = feature_df[feature_df['season'] >= 2010].copy()
        
        # Handle any missing columns
        for feat in features:
            if feat not in recent_data.columns:
                print(f"Warning: Feature {feat} not found in dataset. Skipping.")
        
        X = recent_data[available_features]
        y = recent_data['home_win']
        
        # Split data chronologically (don't shuffle for time series data)
        train_cutoff = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_cutoff], X.iloc[train_cutoff:]
        y_train, y_test = y.iloc[:train_cutoff], y.iloc[train_cutoff:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, grid_search=True):
        """
        Train a machine learning model for game prediction
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training target
        grid_search : bool
            Whether to perform grid search for hyperparameter tuning
        
        Returns:
        --------
        model
            Trained model
        """
        print("Training model...")
        
        if grid_search:
            # Define pipeline with scaling and model
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier())
            ])
            
            # Define hyperparameter grid
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5]
            }
            
            # Perform grid search
            grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            # Get best model
            self.model = grid.best_estimator_
            print(f"Best parameters: {grid.best_params_}")
        else:
            # Use default model without grid search
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5))
            ])
            
            self.model = pipeline.fit(X_train, y_train)
        
        print("Model training complete.")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test target
        
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        if self.model is None:
            print("Model not trained yet.")
            return None
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Evaluate betting performance
        self.evaluate_betting_performance(X_test, y_test)
        
        return metrics
    
    def evaluate_betting_performance(self, X_test, y_test, kelly_fraction=0.25):
        """
        Evaluate betting performance using the model
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test target
        kelly_fraction : float
            Fraction of Kelly criterion to use for bet sizing
        """
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Create a DataFrame for analysis
        betting_df = pd.DataFrame({
            'true_outcome': y_test.values,
            'predicted_prob': y_pred_proba,
            'vegas_prob': X_test['home_implied_prob_normalized'].values if 'home_implied_prob_normalized' in X_test.columns else 0.5
        })
        
        # Initial bankroll
        initial_bankroll = 1000
        current_bankroll = initial_bankroll
        
        # Track bets
        bets = []
        
        for idx, row in betting_df.iterrows():
            # Skip if no edge detected
            edge = row['predicted_prob'] - row['vegas_prob']
            
            if edge <= 0.03:  # Only bet when we have a significant edge
                continue
            
            # Calculate Kelly stake (with fraction for safety)
            expected_value = 2 * row['predicted_prob'] - 1  # EV for fair odds
            kelly_stake = kelly_fraction * expected_value * current_bankroll
            
            # Limit stake to 5% of bankroll
            stake = min(kelly_stake, 0.05 * current_bankroll)
            
            # Record the bet and outcome
            won = row['true_outcome'] == 1
            profit = stake if won else -stake
            current_bankroll += profit
            
            bets.append({
                'prediction': row['predicted_prob'],
                'vegas_prob': row['vegas_prob'],
                'edge': edge,
                'stake': stake,
                'won': won,
                'profit': profit,
                'bankroll': current_bankroll
            })
        
        # Create DataFrame of bets
        if bets:
            bets_df = pd.DataFrame(bets)
            
            # Calculate metrics
            total_bets = len(bets_df)
            winning_bets = bets_df['won'].sum()
            win_rate = winning_bets / total_bets if total_bets > 0 else 0
            roi = (current_bankroll - initial_bankroll) / initial_bankroll
            
            print("\nBetting Performance:")
            print(f"Total bets placed: {total_bets}")
            print(f"Winning bets: {winning_bets} ({win_rate:.2%})")
            print(f"Final bankroll: ${current_bankroll:.2f}")
            print(f"ROI: {roi:.2%}")
            
            # Check performance vs. Vegas by probability bucket
            print("\nPerformance by prediction confidence:")
            bets_df['prob_bucket'] = pd.cut(bets_df['prediction'], bins=[0, 0.55, 0.60, 0.65, 0.70, 1.0])
            bucket_stats = bets_df.groupby('prob_bucket').agg({
                'won': ['count', 'mean'],
                'profit': 'sum'
            })
            print(bucket_stats)
        else:
            print("No bets were placed based on model predictions.")
    
    def feature_importance(self):
        """
        Get feature importances from the model
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with feature importances
        """
        if self.model is None:
            print("Model not trained yet.")
            return None
        
        # Get feature names
        feature_names = self.model.named_steps['classifier'].feature_names_in_
        
        # Get importances
        importances = self.model.named_steps['classifier'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        print("Top 10 most important features:")
        print(importance_df.head(10))
        
        return importance_df
    
    def predict_game(self, home_team, visiting_team, game_date=None, odds=None):
        """
        Predict the outcome of a specific game
        
        Parameters:
        -----------
        home_team : str
            Home team abbreviation
        visiting_team : str
            Visiting team abbreviation
        game_date : str
            Game date in YYYYMMDD format (defaults to today)
        odds : dict
            Dictionary with 'home_moneyline' and 'away_moneyline' keys
        
        Returns:
        --------
        dict
            Dictionary with prediction details
        """
        if self.model is None:
            print("Model not trained yet.")
            return None
        
        if game_date is None:
            game_date = datetime.now().strftime('%Y%m%d')
        
        # Get team historical data
        home_team_data = self.game_data[
            (self.game_data['home_team'] == home_team) | 
            (self.game_data['visiting_team'] == home_team)
        ].sort_values('date')
        
        visiting_team_data = self.game_data[
            (self.game_data['home_team'] == visiting_team) | 
            (self.game_data['visiting_team'] == visiting_team)
        ].sort_values('date')
        
        # Calculate rolling stats for both teams
        window_sizes = [5, 10, 20]
        
        # Create a dictionary to store features
        features = {}
        
        # Process home team stats
        for window in window_sizes:
            # Calculate home team rolling stats
            home_recent_games = home_team_data.tail(window)
            
            # Get games where team is home
            home_games = home_recent_games[home_recent_games['home_team'] == home_team]
            
            # Get games where team is away
            away_games = home_recent_games[home_recent_games['visiting_team'] == home_team]
            
            # Calculate stats
            runs_scored = (
                home_games['home_score'].sum() + 
                away_games['visiting_score'].sum()
            ) / max(1, len(home_recent_games))
            
            runs_allowed = (
                home_games['visiting_score'].sum() + 
                away_games['home_score'].sum()
            ) / max(1, len(home_recent_games))
            
            wins = (
                (home_games['home_score'] > home_games['visiting_score']).sum() +
                (away_games['visiting_score'] > away_games['home_score']).sum()
            )
            
            win_pct = wins / max(1, len(home_recent_games))
            
            features[f'home_rolling_{window}_runs_scored'] = runs_scored
            features[f'home_rolling_{window}_runs_allowed'] = runs_allowed
            features[f'home_rolling_{window}_win_pct'] = win_pct
            features[f'home_rolling_{window}_run_diff'] = runs_scored - runs_allowed
        
        # Process visiting team stats
        for window in window_sizes:
            # Calculate visiting team rolling stats
            visiting_recent_games = visiting_team_data.tail(window)
            
            # Get games where team is home
            home_games = visiting_recent_games[visiting_recent_games['home_team'] == visiting_team]
            
            # Get games where team is away
            away_games = visiting_recent_games[visiting_recent_games['visiting_team'] == visiting_team]
            
            # Calculate stats
            runs_scored = (
                home_games['home_score'].sum() + 
                away_games['visiting_score'].sum()
            ) / max(1, len(visiting_recent_games))
            
            runs_allowed = (
                home_games['visiting_score'].sum() + 
                away_games['home_score'].sum()
            ) / max(1, len(visiting_recent_games))
            
            wins = (
                (home_games['home_score'] > home_games['visiting_score']).sum() +
                (away_games['visiting_score'] > away_games['home_score']).sum()
            )
            
            win_pct = wins / max(1, len(visiting_recent_games))
            
            features[f'visiting_rolling_{window}_runs_scored'] = runs_scored
            features[f'visiting_rolling_{window}_runs_allowed'] = runs_allowed
            features[f'visiting_rolling_{window}_win_pct'] = win_pct
            features[f'visiting_rolling_{window}_run_diff'] = runs_scored - runs_allowed
        
        # Calculate comparison features
        for window in window_sizes:
            features[f'runs_scored_diff_{window}'] = features[f'home_rolling_{window}_runs_scored'] - features[f'visiting_rolling_{window}_runs_scored']
            features[f'runs_allowed_diff_{window}'] = features[f'home_rolling_{window}_runs_allowed'] - features[f'visiting_rolling_{window}_runs_allowed']
            features[f'win_pct_diff_{window}'] = features[f'home_rolling_{window}_win_pct'] - features[f'visiting_rolling_{window}_win_pct']
        
        # Get most recent rest days (simplified approach)
        home_last_game = home_team_data.iloc[-1]['date'] if not home_team_data.empty else None
        visiting_last_game = visiting_team_data.iloc[-1]['date'] if not visiting_team_data.empty else None
        
        if home_last_game and game_date:
            home_days_rest = (pd.to_datetime(game_date) - pd.to_datetime(home_last_game)).days
        else:
            home_days_rest = 1  # Default
        
        if visiting_last_game and game_date:
            visiting_days_rest = (pd.to_datetime(game_date) - pd.to_datetime(visiting_last_game)).days
        else:
            visiting_days_rest = 1  # Default
        
        features['days_rest_advantage'] = home_days_rest - visiting_days_rest
        
        # Add day of week and month features
        game_datetime = pd.to_datetime(game_date)
        day_of_week = game_datetime.dayofweek
        month = game_datetime.month
        
        # Initialize all day and month features to 0
        for i in range(7):
            features[f'day_{i}'] = 0
        for i in range(4, 11):
            features[f'month_{i}'] = 0
        
        # Set the appropriate day and month to 1
        features[f'day_{day_of_week}'] = 1
        features[f'month_{month}'] = 1
        
        # Add odds features if provided
        if odds:
            home_ml = odds.get('home_moneyline', 0)
            away_ml = odds.get('away_moneyline', 0)
            
            # Calculate implied probabilities
            home_prob = (100 / (home_ml + 100)) if home_ml > 0 else (abs(home_ml) / (abs(home_ml) + 100))
            away_prob = (100 / (away_ml + 100)) if away_ml > 0 else (abs(away_ml) / (abs(away_ml) + 100))
            
            # Normalize probabilities
            total_prob = home_prob + away_prob
            home_prob_norm = home_prob / total_prob
            away_prob_norm = away_prob / total_prob
            
            features['home_implied_prob_normalized'] = home_prob_norm
            features['away_implied_prob_normalized'] = away_prob_norm
        else:
            # If odds not provided, use neutral values
            features['home_implied_prob_normalized'] = 0.5
            features['away_implied_prob_normalized'] = 0.5
        
        # Assume night game by default
        features['is_night_game'] = 1
        
        # Create DataFrame with features
        feature_df = pd.DataFrame([features])
        
        # Get subset of features used in the model
        model_features = self.model.feature_names_in_
        available_features = [f for f in model_features if f in feature_df.columns]
        
        # Fill any missing features with 0
        for feat in model_features:
            if feat not in feature_df.columns:
                feature_df[feat] = 0
        
        # Predict win probability
        win_prob = self.model.predict_proba(feature_df[model_features])[0, 1]
        
        # Return prediction details
        return {
            'game_date': game_date,
            'home_team': home_team,
            'visiting_team': visiting_team,
            'home_win_probability': win_prob,
            'visiting_win_probability': 1 - win_prob,
            'prediction': 'Home Win' if win_prob > 0.5 else 'Visiting Win',
            'confidence': max(win_prob, 1 - win_prob)
        }
    
    def run_complete_pipeline(self, years=None, test_size=0.2):
        """
        Run the complete model pipeline from data download to evaluation
        
        Parameters:
        -----------
        years : list
            List of years to include in the analysis (e.g., [2015, 2016, 2017, 2018, 2019])
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Dictionary with model performance metrics
        """
        if years is None:
            years = list(range(2015, 2023))  # Default to recent years
        
        # Download and load data
        self.download_retrosheet_data(years)
        game_data = self.load_retrosheet_data(years)
        
        if game_data is None or len(game_data) == 0:
            print("Failed to load game data. Aborting pipeline.")
            return None
        
        # Get odds data (placeholder for real implementation)
        odds_data = self.download_odds_data(years)
        
        # Merge data
        merged_data = self.merge_game_and_odds_data()
        
        if merged_data is None or len(merged_data) == 0:
            print("Failed to merge game and odds data. Aborting pipeline.")
            return None
        
        # Engineer features
        feature_df = self.engineer_features(merged_data)
        
        # Prepare data for modeling
        X_train, X_test, y_train, y_test = self.prepare_model_data(feature_df)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Get feature importance
        importance_df = self.feature_importance()
        
        # Return performance metrics
        return metrics


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    mlb_predictor = MLBPredictor()
    
    # Run pipeline with data from 2018-2022
    years = list(range(2018, 2023))
    metrics = mlb_predictor.run_complete_pipeline(years)
    
    # Example: Predict a specific game
    prediction = mlb_predictor.predict_game(
        home_team="NYA",    # New York Yankees
        visiting_team="BOS", # Boston Red Sox
        game_date="20230601", # June 1, 2023
        odds={
            'home_moneyline': -150,  # Yankees favored
            'away_moneyline': +130   # Red Sox underdogs
        }
    )
    
    print("\nGame Prediction:")
    for key, value in prediction.items():
        print(f"{key}: {value}")