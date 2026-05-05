import pandas as pd
import joblib
import os
from src.preprocess import preprocess_data

def run_simulation():
    """
    Simulates predicting the performance of new students.
    """
    print("\n--- Running Virtual Simulation ---")
    
    # Define 5 virtual students
    virtual_students = pd.DataFrame({
        "Study_Hours": [10.5, 35.0, 20.0, 5.0, 25.0],
        "Attendance": [75.0, 98.0, 85.0, 50.0, 92.0],
        "Previous_Score": [60.0, 95.0, 78.0, 40.0, 88.0],
        "Extracurricular": [0, 1, 1, 0, 1],
        "Sleep_Hours": [6.5, 8.0, 7.5, 5.0, 7.0],
        "Internet_Access": [1, 1, 1, 0, 1]
    })
    
    # We add dummy columns for the drop/preprocess step to work exactly the same way if needed
    # But since preprocess_data handles missing target cols by ignoring them, we can just pass this
    
    try:
        model = joblib.load("models/rf_model.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
    except FileNotFoundError:
        print("Model or feature names not found. Please train the model first.")
        return
        
    # Ensure columns match the feature names order (they should by default here)
    virtual_students = virtual_students[list(feature_names)]
    
    # Preprocess
    X_scaled = preprocess_data(virtual_students, is_training=False)
    
    # Predict
    predictions = model.predict(X_scaled)
    
    # Combine with input data for output
    results = virtual_students.copy()
    results["Predicted_Final_Score"] = [round(p, 1) for p in predictions]
    
    # Add Risk Level
    # If predicted score < 50: High Risk
    # If predicted score < 70: Medium Risk
    # Else: Low Risk
    
    def get_risk_level(score):
        if score < 50: return "High Risk (Likely to Fail)"
        elif score < 70: return "Medium Risk"
        else: return "Low Risk (Safe)"
        
    results["Risk_Level"] = results["Predicted_Final_Score"].apply(get_risk_level)
    
    print("\nSimulation Results for 5 New Students:")
    print("-" * 60)
    print(results.to_string(index=False))
    
    # Save simulation results
    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/simulation_results.csv", index=False)
    print("\nSimulation results saved to outputs/simulation_results.csv")
    
if __name__ == "__main__":
    run_simulation()
