"""
Data loading module for MLB prediction model.

This module handles downloading Retrosheet game logs, loading historical game data,
generating/downloading odds data, and merging game and odds datasets.

TODO: Replace placeholder odds generation with real historical odds API integration
TODO: Add support for additional data sources (weather, injuries, starting pitchers)
"""

import pandas as pd
import numpy as np
import requests
import os
import zipfile
import io
from typing import List, Optional


# Retrosheet game log column mapping based on official documentation
# Source: https://www.retrosheet.org/gamelogs/glfields.txt
RETROSHEET_COLUMNS = {
    0: 'date',
    3: 'visiting_team',
    6: 'home_team',
    9: 'visiting_score',
    10: 'home_score',
    12: 'day_night',
    # Box score stats for home team
    48: 'home_AB',  # At-bats
    49: 'home_H',   # Hits
    50: 'home_2B',  # Doubles
    51: 'home_3B',  # Triples
    52: 'home_HR',  # Home runs
    # Box score stats for visiting team
    25: 'visiting_AB',  # At-bats
    26: 'visiting_H',   # Hits
    27: 'visiting_2B',  # Doubles
    28: 'visiting_3B',  # Triples
    29: 'visiting_HR',  # Home runs
}


def download_retrosheet_data(years: List[int], data_dir: str = './data') -> None:
    """
    Download game data from Retrosheet for specified years.
    
    Parameters
    ----------
    years : list of int
        Years to download data for (e.g., [2018, 2019, 2020])
    data_dir : str, optional
        Directory to store downloaded data
        
    Notes
    -----
    Retrosheet provides complete game logs in a standardized format.
    Files are downloaded as .zip and extracted automatically.
    """
    print(f"Downloading Retrosheet data for years: {years}")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for year in years:
        url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(data_dir)
                print(f"Successfully downloaded and extracted data for {year}")
            else:
                print(f"Failed to download data for {year}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading data for {year}: {e}")


def load_retrosheet_data(years: List[int], data_dir: str = './data') -> Optional[pd.DataFrame]:
    """
    Load and parse Retrosheet game logs.
    
    Parameters
    ----------
    years : list of int
        Years to load data for
    data_dir : str, optional
        Directory containing downloaded data files
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame containing game data with properly mapped columns,
        or None if no data could be loaded
        
    Notes
    -----
    The function maps Retrosheet columns to meaningful names based on the
    official field specification. This fixes the KeyError issue by ensuring
    batting statistics columns (home_H, home_AB, etc.) are properly created.
    
    TODO: Add validation to check for corrupt or incomplete game log files
    TODO: Consider caching parsed data to avoid re-parsing on subsequent runs
    """
    all_data = []
    
    for year in years:
        file_path = os.path.join(data_dir, f"GL{year}.TXT")
        if os.path.exists(file_path):
            try:
                # Load data without predefined column names
                year_data = pd.read_csv(file_path, header=None, sep=',', quotechar='"')
                
                # Create generic column names first
                num_columns = year_data.shape[1]
                column_names = [f'col_{i}' for i in range(num_columns)]
                year_data.columns = column_names
                
                # Map known columns based on Retrosheet specification
                rename_map = {f'col_{idx}': name for idx, name in RETROSHEET_COLUMNS.items()}
                year_data.rename(columns=rename_map, inplace=True)
                
                # Add year column for tracking
                year_data['season'] = year
                
                # Ensure batting stat columns exist (defensive programming)
                stat_columns = ['home_H', 'home_AB', 'visiting_H', 'visiting_AB',
                               'home_2B', 'home_3B', 'home_HR', 
                               'visiting_2B', 'visiting_3B', 'visiting_HR']
                
                for col in stat_columns:
                    if col not in year_data.columns:
                        print(f"Warning: {col} not found for {year}, setting to 0")
                        year_data[col] = 0
                
                # Convert numeric columns to appropriate types
                numeric_cols = ['visiting_score', 'home_score', 'home_H', 'home_AB',
                               'visiting_H', 'visiting_AB', 'home_2B', 'home_3B', 
                               'home_HR', 'visiting_2B', 'visiting_3B', 'visiting_HR']
                
                for col in numeric_cols:
                    if col in year_data.columns:
                        year_data[col] = pd.to_numeric(year_data[col], errors='coerce').fillna(0)
                
                all_data.append(year_data)
                print(f"Loaded {len(year_data)} games from {year}")
                
            except Exception as e:
                print(f"Error loading data for {year}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    if all_data:
        game_data = pd.concat(all_data, ignore_index=True)
        print(f"Total games loaded: {len(game_data)}")
        return game_data
    else:
        print("No data was loaded.")
        return None


def download_odds_data(years: List[int]) -> pd.DataFrame:
    """
    Generate placeholder odds data (to be replaced with real API integration).
    
    Parameters
    ----------
    years : list of int
        Years to generate odds data for
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, home_team, visiting_team, 
        home_moneyline, away_moneyline
        
    Notes
    -----
    This is a PLACEHOLDER function that generates synthetic odds data.
    
    TODO: Replace with actual historical odds data from one of these sources:
        - The Odds API (https://the-odds-api.com/) - provides historical odds
        - SportsOddsHistory.com - historical odds database
        - OddsPortal scraping (check terms of service)
        - Purchase historical data from a sports data provider
    
    OVERSIMPLIFICATION WARNING:
        The current random odds generation:
        - Does not reflect actual market conditions
        - Ignores home field advantage patterns
        - Doesn't account for pitcher matchups
        - Has no correlation with actual team strength
        
    For production use, this MUST be replaced with real historical odds.
    """
    print("Note: Generating placeholder odds data.")
    print("TODO: Replace with actual historical odds API integration")
    
    dates = []
    home_teams = []
    away_teams = []
    home_moneyline = []
    away_moneyline = []
    
    # MLB team abbreviations (Retrosheet format)
    teams = ['NYA', 'BOS', 'TOR', 'BAL', 'TBA',
             'CHA', 'CLE', 'DET', 'KCA', 'MIN',
             'HOU', 'LAA', 'OAK', 'SEA', 'TEX',
             'ATL', 'MIA', 'NYN', 'PHI', 'WAS',
             'CHN', 'CIN', 'MIL', 'PIT', 'SLN',
             'ARI', 'COL', 'LAN', 'SDN', 'SFN']
    
    for year in years:
        for month in range(4, 11):  # Baseball season (April-October)
            for day in range(1, 28):
                if np.random.random() < 0.3:  # Not every day has games
                    continue
                
                # Generate 8 random games for this date
                for _ in range(8):
                    date_str = f"{year}{month:02d}{day:02d}"
                    game_teams = np.random.choice(teams, 2, replace=False)
                    home_team = game_teams[0]
                    away_team = game_teams[1]
                    
                    # Generate random odds
                    # TODO: Model this based on actual betting market patterns
                    is_home_favorite = np.random.random() > 0.4
                    
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
    
    odds_data = pd.DataFrame({
        'date': dates,
        'home_team': home_teams,
        'visiting_team': away_teams,
        'home_moneyline': home_moneyline,
        'away_moneyline': away_moneyline
    })
    
    print(f"Created placeholder odds data with {len(odds_data)} entries")
    return odds_data


def merge_game_and_odds_data(game_data: pd.DataFrame, 
                             odds_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Merge game data with odds data on date and team matchup.
    
    Parameters
    ----------
    game_data : pd.DataFrame
        Game results data from Retrosheet
    odds_data : pd.DataFrame
        Betting odds data
    
    Returns
    -------
    pd.DataFrame or None
        Merged dataset, or None if inputs are invalid
        
    Notes
    -----
    Uses inner join to ensure we only keep games where both actual results
    and odds are available. This is important for training since we need
    both features (odds) and labels (results).
    
    TODO: Add logic to handle multiple odds from different sportsbooks
    TODO: Consider keeping games without odds for evaluation purposes
    """
    if game_data is None or odds_data is None:
        print("Cannot merge: game_data or odds_data is None")
        return None
    
    # Convert date formats to match
    game_data['date_str'] = game_data['date'].astype(str)
    
    # Merge on date and team names
    merged_data = pd.merge(
        game_data,
        odds_data,
        left_on=['date_str', 'home_team', 'visiting_team'],
        right_on=['date', 'home_team', 'visiting_team'],
        how='inner'
    )
    
    print(f"Merged data has {len(merged_data)} rows")
    print(f"Match rate: {len(merged_data) / len(game_data) * 100:.1f}% of games have odds")
    
    return merged_data
