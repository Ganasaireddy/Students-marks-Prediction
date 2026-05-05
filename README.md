# 🎓 Student Performance Prediction System

An end-to-end Machine Learning project to predict student academic performance and identify at-risk students based on simulated behavioral and demographic data. 

This project was built to demonstrate a complete Data Science pipeline—from synthetic data generation and preprocessing to model training, evaluation, and virtual simulation.

## 🎯 Project Overview

**What is Student Performance Prediction?**
Educational institutions collect vast amounts of data (attendance, study hours, past scores, participation). By using Machine Learning, we can analyze this data to predict a student's final score automatically.

**Why is it important?**
- **Identifying Weak Students:** Early warning systems flag students likely to score poorly.
- **Dropout Prevention:** Identifying behavioral patterns that lead to dropouts and intervening early.
- **Personalized Learning:** Recommending specific study paths based on predicted weaknesses.

**Data Workflow:**
`Student Data (Synthetic)` → `Preprocessing (Cleaning & Scaling)` → `Model Training (Random Forest)` → `Prediction` → `Actionable Insights`

---

## 🛠️ Tech Stack & Tools

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest Regressor)
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib

---

## 📂 Folder Structure

```text
Student-Performance-Prediction/
│
├── data/                  # Contains generated student_data.csv (1000+ records)
├── notebooks/             # Directory for exploratory Jupyter notebooks
├── src/                   # Core Python modules
│   ├── data_generator.py  # Generates realistic synthetic student data
│   ├── preprocess.py      # Cleans, splits, and scales data
│   ├── model.py           # Trains and evaluates the Random Forest model
│   └── simulate.py        # Runs virtual predictions on new student profiles
├── models/                # Saved trained models (scaler.pkl, rf_model.pkl)
├── outputs/               # Evaluation metrics and simulation CSVs
├── images/                # Visualizations (Feature Importance, Scatter Plots)
├── requirements.txt       # Python dependencies
└── main.py                # Master script to run the full pipeline
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run the Project

Run the entire pipeline automatically using the `main.py` script:

```bash
python main.py
```

**What happens when you run `main.py`?**
1. **Phase 1:** Generates a synthetic dataset of 1,000 students (if it doesn't already exist).
2. **Phase 2:** Preprocesses the data and saves the standard scaler.
3. **Phase 3:** Trains a Random Forest Regressor and saves the model.
4. **Phase 4:** Evaluates the model (MAE, RMSE, R2) and generates visualization plots in the `images/` folder.
5. **Phase 5:** Runs a "Virtual Simulation" on 5 new fictional students to predict their final scores and assess their risk level.

---

## 📊 Virtual Simulation

The project includes a `simulate.py` module that acts as a real-world engine. Given a new student's data (e.g., Study Hours: 5, Attendance: 50%, Previous Score: 40), the pre-trained model will predict their Final Score and assign a Risk Level (e.g., "High Risk", "Medium Risk", "Low Risk"). 

Check `outputs/simulation_results.csv` after running the project to see the simulation output.

---

## 📸 Screenshots & Outputs

*(Upload your generated images from the `images/` folder and link them here)*

- **Feature Importance Plot:** Shows which factors (e.g., Attendance, Study Hours) influence the final score the most. (`images/feature_importance.png`)
- **Actual vs Predicted Plot:** Demonstrates the accuracy of the Random Forest model. (`images/actual_vs_predicted.png`)

---

## 📅 5-Day GitHub Proof Strategy

If you are using this to build your portfolio, follow this commit strategy to show organic progress:

- **Day 1:** Project setup, `requirements.txt`, and basic `README.md`.
- **Day 2:** Build `src/data_generator.py` and commit the synthetic dataset.
- **Day 3:** Build `src/preprocess.py` and perform data scaling/cleaning.
- **Day 4:** Implement `src/model.py`, train the model, and generate evaluation metrics/plots.
- **Day 5:** Add `src/simulate.py`, create `main.py`, and upload the final `README.md` with images.

---
*Developed for Data Science & Machine Learning Portfolio Building.*
