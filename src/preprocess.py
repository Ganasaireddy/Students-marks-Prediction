import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def load_data(filepath="data/student_data.csv"):
    """Loads the dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df, target_col="Final_Score", is_training=True):
    """
    Preprocesses the data: drops unnecessary columns, handles scaling.
    If is_training is True, splits data and fits the scaler.
    If is_training is False, just applies the scaler.
    """
    print("Preprocessing data...")
    
    # Drop Student_ID and Status (since we are doing regression on Final_Score)
    # We drop 'Status' because it's directly derived from Final_Score
    cols_to_drop = ["Student_ID", "Status"]
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)
        
    X = df.drop(columns=cols_to_drop, errors='ignore')
    
    if is_training:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
            
        y = df[target_col]
        X = X.drop(columns=[target_col], errors='ignore')
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler for future predictions
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        print("Scaler saved to models/scaler.pkl")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    else:
        # Load scaler
        try:
            scaler = joblib.load("models/scaler.pkl")
            X_scaled = scaler.transform(X)
            return X_scaled
        except FileNotFoundError:
            raise FileNotFoundError("Scaler not found. Please train the model first.")

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
