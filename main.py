"""
MLB Game Prediction Model - Main Entry Point

This script demonstrates the complete workflow:
1. Train model on historical data
2. Evaluate model performance
3. Demonstrate single game prediction

Run with: python main.py
"""
from predictor import MLBPredictor
from predict import format_prediction_output


def main():
    """
    Run the complete MLB prediction pipeline.
    """
    # Initialize predictor
    print("Initializing MLB Predictor...")
    mlb_predictor = MLBPredictor(data_dir='./data')
    
    # Define years to train on
    # Using 2018-2022 for demonstration
    # For production, use more years for better model
    years = list(range(2018, 2023))
    
    # Run complete pipeline
    print(f"\nRunning pipeline for years {years[0]}-{years[-1]}...")
    results = mlb_predictor.run_complete_pipeline(
        years=years,
        test_size=0.2,
        grid_search=True,  # Set to False for faster training (uses defaults)
        evaluate_betting=True
    )
    
    # Display summary of results
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Model accuracy:       {results['metrics']['accuracy']:.3f}")
    print(f"ROC AUC:              {results['metrics']['roc_auc']:.3f}")
    print(f"Brier score:          {results['metrics']['brier_score']:.4f}")
    
    if results['betting_results']:
        print(f"\nBetting Performance:")
        print(f"Total bets:           {results['betting_results']['total_bets']}")
        print(f"Win rate:             {results['betting_results']['win_rate']:.1%}")
        print(f"ROI:                  {results['betting_results']['roi']:.1%}")
        print(f"Final bankroll:       ${results['betting_results']['final_bankroll']:.2f}")
    
    print("=" * 70)
    
    # Demonstrate game prediction
    print("\n\nDEMONSTRATING SINGLE GAME PREDICTION")
    print("=" * 70)
    
    # Example prediction: Yankees vs Red Sox
    prediction = mlb_predictor.predict_game(
        home_team="NYA",    # New York Yankees
        visiting_team="BOS", # Boston Red Sox
        game_date="20230601", # June 1, 2023
        odds={
            'home_moneyline': -150,  # Yankees favored
            'away_moneyline': +130   # Red Sox underdogs
        }
    )
    
    # Display formatted prediction
    print(format_prediction_output(prediction))
    
    print("\n✅ Pipeline completed successfully!")



if __name__ == "__main__":
    main()
