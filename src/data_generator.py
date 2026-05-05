import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_students=1000, output_path="data/student_data.csv"):
    """
    Generates synthetic student performance data.
    """
    np.random.seed(42)
    
    print(f"Generating synthetic data for {num_students} students...")
    
    # Generate features
    student_id = np.arange(1, num_students + 1)
    
    # Study hours per week (normal distribution centered around 15, range 2 to 40)
    study_hours = np.clip(np.random.normal(15, 8, num_students), 2, 40)
    
    # Attendance percentage (normal distribution centered around 85%, range 40 to 100)
    attendance = np.clip(np.random.normal(85, 15, num_students), 40, 100)
    
    # Previous scores (out of 100)
    previous_score = np.clip(np.random.normal(70, 15, num_students), 20, 100)
    
    # Participation in extracurricular activities (binary)
    extracurricular = np.random.choice([0, 1], size=num_students, p=[0.6, 0.4])
    
    # Sleep hours per night
    sleep_hours = np.clip(np.random.normal(7, 1.5, num_students), 4, 10)
    
    # Internet access at home (binary)
    internet_access = np.random.choice([0, 1], size=num_students, p=[0.1, 0.9])
    
    # Generate the target variable: Final Score
    # We create a linear relationship with some noise
    # Base score
    final_score = 10 + (study_hours * 0.8) + (attendance * 0.4) + (previous_score * 0.3) + (extracurricular * 3) + (sleep_hours * 1.5) + (internet_access * 2)
    
    # Add random noise
    noise = np.random.normal(0, 5, num_students)
    final_score += noise
    
    # Clip final score to max 100 and min 0
    final_score = np.clip(final_score, 0, 100)
    
    # Determine Pass/Fail (Pass if score >= 50)
    status = np.where(final_score >= 50, "Pass", "Fail")
    
    # Create DataFrame
    df = pd.DataFrame({
        "Student_ID": student_id,
        "Study_Hours": np.round(study_hours, 1),
        "Attendance": np.round(attendance, 1),
        "Previous_Score": np.round(previous_score, 1),
        "Extracurricular": extracurricular,
        "Sleep_Hours": np.round(sleep_hours, 1),
        "Internet_Access": internet_access,
        "Final_Score": np.round(final_score, 1),
        "Status": status
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()
