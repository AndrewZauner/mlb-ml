"""
Betting performance module for MLB prediction model.

This module simulates betting strategies and evaluates profitability.

TODO: Implement proper Kelly Criterion with risk management
TODO: Add closing line value (CLV) tracking
TODO: Include vig/juice calculations for different sportsbooks
TODO: Add bankroll management and drawdown analysis
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import Dict, Optional


def evaluate_betting_performance(model: Pipeline,
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series,
                                 kelly_fraction: float = 0.25,
                                 min_edge: float = 0.03,
                                 max_bet_pct: float = 0.05,
                                 initial_bankroll: float = 1000.0) -> Dict:
    """
    Evaluate betting performance using the model's predictions.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained prediction model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test outcomes (1 = home win, 0 = away win)
    kelly_fraction : float, optional
        Fraction of Kelly Criterion to use (default: 0.25 for quarter-Kelly)
    min_edge : float, optional
        Minimum edge required to place bet (default: 0.03 = 3%)
    max_bet_pct : float, optional
        Maximum bet as fraction of bankroll (default: 0.05 = 5%)
    initial_bankroll : float, optional
        Starting bankroll (default: 1000.0)
    
    Returns
    -------
    dict
        Dictionary with betting performance metrics
        
    Notes
    -----
    CURRENT BETTING STRATEGY:
    - Bet when model probability > vegas probability by at least min_edge
    - Use fractional Kelly for bet sizing
    - Cap bets at max_bet_pct of bankroll for risk management
    - Assume fair odds (2.0 decimal = +100 American)
    
    OVERSIMPLIFICATIONS IN CURRENT IMPLEMENTATION:
    
    1. ODDS STRUCTURE:
       - Assumes fair odds without vig
       - Reality: Sportsbooks charge -110 on both sides (~4.5% vig)
       - TODO: Model actual odds with vig, find best available lines
    
    2. BET SIZING:
       - Uses simplified Kelly with fair odds
       - TODO: Implement proper Kelly with actual American odds
       - Kelly formula: f = (bp - q) / b
         where: f = fraction to bet
                b = odds received (decimal - 1)
                p = probability of winning
                q = 1 - p
    
    3. BANKROLL MANAGEMENT:
       - No stop-loss or take-profit rules
       - No adjustment for drawdowns
       - TODO: Implement risk of ruin calculations
       - TODO: Add dynamic Kelly fraction based on confidence
    
    4. MARKET EFFICIENCY:
       - Assumes we can always bet at listed odds
       - Reality: Lines move, limits exist, odds can be pulled
       - TODO: Model closing line value (CLV) to validate edge
    
    5. FLAT BETTING COMPARISON:
       - TODO: Compare Kelly to flat betting (same $ each bet)
       - Flat betting is simpler and often safer
    
    6. MULTIPLE SPORTSBOOKS:
       - TODO: Shop for best lines across books
       - Line shopping can add 1-3% to ROI
    
    7. CORRELATION:
       - Treats each bet independently
       - Reality: Some games are correlated (division rivals, weather)
       - TODO: Consider portfolio effects
    """
    print("\n" + "=" * 70)
    print("EVALUATING BETTING PERFORMANCE")
    print("=" * 70)
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create betting analysis DataFrame
    betting_df = pd.DataFrame({
        'true_outcome': y_test.values,
        'predicted_prob': y_pred_proba,
        'vegas_prob': X_test['home_implied_prob_normalized'].values 
                      if 'home_implied_prob_normalized' in X_test.columns else 0.5
    })
    
    # Initialize tracking
    current_bankroll = initial_bankroll
    bets = []
    
    print(f"\nInitial Bankroll: ${initial_bankroll:.2f}")
    print(f"Kelly Fraction: {kelly_fraction:.2f} (conservative)")
    print(f"Minimum Edge: {min_edge:.1%}")
    print(f"Maximum Bet: {max_bet_pct:.1%} of bankroll\n")
    
    # Simulate betting on each game
    for idx, row in betting_df.iterrows():
        # Calculate edge (our probability - market probability)
        edge = row['predicted_prob'] - row['vegas_prob']
        
        # Only bet when we have sufficient edge
        if edge <= min_edge:
            continue
        
        # SIMPLIFIED KELLY CRITERION
        # This assumes fair odds (decimal odds of 2.0)
        # TODO: Calculate with actual American odds from sportsbook
        expected_value = 2 * row['predicted_prob'] - 1
        kelly_stake = kelly_fraction * expected_value * current_bankroll
        
        # Apply maximum bet constraint (risk management)
        stake = min(kelly_stake, max_bet_pct * current_bankroll)
        
        # Ensure positive stake
        if stake <= 0:
            continue
        
        # Determine outcome
        won = row['true_outcome'] == 1
        
        # Calculate profit/loss (assumes fair odds)
        # TODO: Use actual odds to calculate returns
        profit = stake if won else -stake
        current_bankroll += profit
        
        # Record bet
        bets.append({
            'prediction': row['predicted_prob'],
            'vegas_prob': row['vegas_prob'],
            'edge': edge,
            'stake': stake,
            'stake_pct': stake / (current_bankroll - profit),  # % of bankroll before bet
            'won': won,
            'profit': profit,
            'bankroll': current_bankroll
        })
    
    # Analyze results
    if not bets:
        print("⚠️  No bets were placed based on criteria.")
        print(f"   Try lowering min_edge (currently {min_edge:.1%})")
        return {
            'total_bets': 0,
            'final_bankroll': initial_bankroll,
            'roi': 0.0
        }
    
    bets_df = pd.DataFrame(bets)
    
    # Calculate key metrics
    total_bets = len(bets_df)
    winning_bets = bets_df['won'].sum()
    win_rate = winning_bets / total_bets
    total_staked = bets_df['stake'].sum()
    total_profit = bets_df['profit'].sum()
    roi = (current_bankroll - initial_bankroll) / initial_bankroll
    
    # Profit per bet
    avg_profit_per_bet = bets_df['profit'].mean()
    
    # Maximum drawdown
    bets_df['cumulative_profit'] = bets_df['profit'].cumsum()
    bets_df['running_max'] = bets_df['cumulative_profit'].cummax()
    bets_df['drawdown'] = bets_df['running_max'] - bets_df['cumulative_profit']
    max_drawdown = bets_df['drawdown'].max()
    max_drawdown_pct = max_drawdown / initial_bankroll
    
    # Print summary
    print("=" * 70)
    print("BETTING RESULTS")
    print("=" * 70)
    print(f"Total bets placed:        {total_bets:>8}")
    print(f"Winning bets:             {winning_bets:>8} ({win_rate:.1%})")
    print(f"Losing bets:              {total_bets - winning_bets:>8}")
    print(f"\nTotal staked:             ${total_staked:>8.2f}")
    print(f"Total profit:             ${total_profit:>8.2f}")
    print(f"Average profit/bet:       ${avg_profit_per_bet:>8.2f}")
    print(f"\nFinal bankroll:           ${current_bankroll:>8.2f}")
    print(f"Return on Investment:     {roi:>8.1%}")
    print(f"Max drawdown:             ${max_drawdown:>8.2f} ({max_drawdown_pct:.1%})")
    print("=" * 70)
    
    # Performance by confidence level
    print("\nPerformance by Prediction Confidence:")
    print("-" * 70)
    bets_df['confidence_bucket'] = pd.cut(
        bets_df['prediction'], 
        bins=[0, 0.55, 0.60, 0.65, 0.70, 1.0],
        labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+']
    )
    
    bucket_stats = bets_df.groupby('confidence_bucket', observed=True).agg({
        'won': ['count', 'sum', 'mean'],
        'profit': 'sum',
        'stake': 'sum'
    })
    bucket_stats.columns = ['Bets', 'Wins', 'Win%', 'Profit', 'Staked']
    bucket_stats['ROI%'] = (bucket_stats['Profit'] / bucket_stats['Staked'] * 100).round(1)
    
    print(bucket_stats.to_string())
    print("-" * 70)
    
    # TODO: Add more analysis
    # - Streak analysis (longest winning/losing streak)
    # - Monthly performance
    # - Performance by edge size
    # - Sharpe ratio (risk-adjusted returns)
    
    print("\n💡 IMPORTANT NOTES:")
    print("   • Results assume fair odds without vig")
    print("   • Real sportsbooks typically charge -110 on both sides")
    print("   • This would reduce ROI by approximately 4-5%")
    print("   • Always shop for best lines across multiple books")
    print("   • Track closing line value (CLV) to validate your edge")
    
    # Return metrics
    return {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'total_profit': total_profit,
        'final_bankroll': current_bankroll,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_profit_per_bet': avg_profit_per_bet,
        'performance_by_confidence': bucket_stats
    }


def calculate_expected_value(model_prob: float,
                             market_prob: float,
                             stake: float = 1.0,
                             odds: float = 2.0) -> float:
    """
    Calculate expected value of a bet.
    
    Parameters
    ----------
    model_prob : float
        Our estimated probability of winning
    market_prob : float
        Market's implied probability
    stake : float, optional
        Amount to bet (default: 1.0)
    odds : float, optional
        Decimal odds (default: 2.0 = even money)
    
    Returns
    -------
    float
        Expected value of the bet
        
    Notes
    -----
    EV = (probability of winning × amount won per bet) - 
         (probability of losing × amount lost per bet)
    
    Positive EV indicates a profitable bet in the long run.
    
    TODO: Convert American odds to decimal odds properly
    TODO: Account for push scenarios (ties) in some bet types
    """
    amount_won = stake * (odds - 1)
    amount_lost = stake
    
    ev = (model_prob * amount_won) - ((1 - model_prob) * amount_lost)
    
    return ev


def kelly_criterion(win_prob: float,
                    odds: float,
                    fraction: float = 1.0) -> float:
    """
    Calculate Kelly Criterion bet size.
    
    Parameters
    ----------
    win_prob : float
        Probability of winning the bet (0 to 1)
    odds : float
        Decimal odds received on win
    fraction : float, optional
        Fraction of Kelly to bet (default: 1.0 = full Kelly)
        Use 0.25-0.5 for fractional Kelly (more conservative)
    
    Returns
    -------
    float
        Fraction of bankroll to bet (0 to 1)
        
    Notes
    -----
    Kelly formula: f = (bp - q) / b
    where:
        f = fraction of bankroll to wager
        b = odds received (decimal odds - 1)
        p = probability of winning
        q = 1 - p (probability of losing)
    
    Kelly Criterion maximizes long-term growth rate but can be
    aggressive. Fractional Kelly (0.25 to 0.5) is recommended.
    
    TODO: Implement for American odds format
    TODO: Add safety checks (negative Kelly = don't bet)
    TODO: Consider Kelly for multiple simultaneous bets
    """
    b = odds - 1  # Net odds received
    p = win_prob
    q = 1 - p
    
    # Kelly formula
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly
    kelly_fractional = kelly * fraction
    
    # Never bet more than 100% of bankroll
    kelly_fractional = max(0, min(kelly_fractional, 1.0))
    
    return kelly_fractional
