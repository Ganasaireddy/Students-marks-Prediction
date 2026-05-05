import os
from src.data_generator import generate_synthetic_data
from src.preprocess import load_data, preprocess_data
from src.model import train_model, evaluate_model
from src.simulate import run_simulation

def main():
    print("="*50)
    print("Student Performance Prediction System")
    print("="*50)
    
    # 1. Data Generation
    print("\n[Phase 1] Data Generation")
    data_path = "data/student_data.csv"
    if not os.path.exists(data_path):
        df = generate_synthetic_data(num_students=1000, output_path=data_path)
    else:
        print("Data already exists. Skipping generation.")
        df = load_data(data_path)
        
    # 2. Preprocessing
    print("\n[Phase 2] Data Preprocessing")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # 3. Model Training
    print("\n[Phase 3] Model Training")
    model = train_model(X_train, y_train, feature_names)
    
    # 4. Evaluation
    print("\n[Phase 4] Model Evaluation")
    evaluate_model(model, X_test, y_test, feature_names)
    
    # 5. Virtual Simulation
    print("\n[Phase 5] Virtual Simulation")
    run_simulation()
    
    print("\n" + "="*50)
    print("Pipeline Execution Complete!")
    print("Outputs can be found in the following folders:")
    print(" - models/ (Saved trained models)")
    print(" - images/ (Plots and Visualizations)")
    print(" - outputs/ (Text metrics and simulation results)")
    print("="*50)

if __name__ == "__main__":
    main()
