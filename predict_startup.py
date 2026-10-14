"""
Inference script for predicting startup profit using the trained CRISP-DM MLR pipeline.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import joblib
import pandas as pd

def predict_startup_profit(rd_spend: float, admin_spend: float, marketing_spend: float, state: str) -> float:
    model_path = os.path.join(os.path.dirname(__file__), "startup_profit_pipeline.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained pipeline model not found at {model_path}. Please run train_crisp_dm.py first.")
    
    pipeline = joblib.load(model_path)
    input_df = pd.DataFrame([{
        'R&D Spend': float(rd_spend),
        'Administration': float(admin_spend),
        'Marketing Spend': float(marketing_spend),
        'State': str(state)
    }])
    
    prediction = pipeline.predict(input_df)[0]
    return prediction

if __name__ == "__main__":
    print("-" * 60)
    print("50 Startups Profit Prediction Service (Phase 6: Deployment)")
    print("-" * 60)
    
    # Example queries
    sample_queries = [
        {"R&D Spend": 165000.0, "Administration": 135000.0, "Marketing Spend": 470000.0, "State": "New York"},
        {"R&D Spend": 100000.0, "Administration": 120000.0, "Marketing Spend": 250000.0, "State": "California"},
        {"R&D Spend": 45000.0, "Administration": 90000.0, "Marketing Spend": 150000.0, "State": "Florida"}
    ]
    
    for idx, query in enumerate(sample_queries, 1):
        profit = predict_startup_profit(
            rd_spend=query["R&D Spend"],
            admin_spend=query["Administration"],
            marketing_spend=query["Marketing Spend"],
            state=query["State"]
        )
        print(f"Sample {idx}:")
        print(f"  Inputs: State={query['State']}, R&D=${query['R&D Spend']:,.2f}, Admin=${query['Administration']:,.2f}, Mkt=${query['Marketing Spend']:,.2f}")
        print(f"  --> Predicted Profit: ${profit:,.2f}\n")
