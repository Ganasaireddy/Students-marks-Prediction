import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(X_train, y_train, feature_names):
    """
    Trains a Random Forest Regressor and saves the model.
    """
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_model.pkl")
    print("Model saved to models/rf_model.pkl")
    
    # Save feature names
    joblib.dump(feature_names, "models/feature_names.pkl")
    
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluates the model and generates visualization plots.
    """
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    # Save metrics to outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_metrics.txt", "w") as f:
        f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R2): {r2:.2f}\n")
    
    # Visualizations
    os.makedirs("images", exist_ok=True)
    
    # 1. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Final Score')
    plt.ylabel('Predicted Final Score')
    plt.title('Actual vs Predicted Final Scores')
    plt.tight_layout()
    plt.savefig('images/actual_vs_predicted.png')
    plt.close()
    
    # 2. Feature Importance Plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features, palette='viridis', legend=False)
    plt.title('Feature Importances for Predicting Final Score')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png')
    plt.close()
    
    print("Visualizations saved to images/ folder.")
    return predictions
