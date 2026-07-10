<div align="center">
  
# Cardiovascular Disease Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.0%2B-blue.svg?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-red.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

*An End-to-End Data Mining and Machine Learning pipeline to predict cardiovascular disease risk using XGBoost, Hyperparameter Tuning, and SHAP Explainability.*

</div>

<br />

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [How to Run](#how-to-run)
- [License](#license)

---

## Project Overview

This project aims to predict the probability of a patient developing **Cardiovascular Disease** based on their medical history and health indicators. By applying a robust Data Mining approach (based on the CRISP-DM framework), this repository demonstrates industry-standard practices for cleaning medical data, engineering new features, selecting significant variables, and training a highly optimized predictive model.

**Objective:** 
To accurately classify patients into Healthy (0) or Sick (1) using an **XGBoost Classifier** optimized via **RandomizedSearchCV**, and to explain the model's decision-making process using **SHAP (SHapley Additive exPlanations)**.

---

## Repository Structure

```text
.
 ┣ dataset/                                  # Directory for the cardiovascular dataset (ignored in git)
 ┣ model/                                    # Directory for exported .pkl models (ignored in git)
 ┣ Cardiovascular_Disease_Prediction.ipynb   # Main Jupyter Notebook (End-to-End Code)
 ┣ requirements.txt                          # Python dependencies list
 ┣ .gitignore                                # Git ignore configurations
 ┗ README.md                                 # Project documentation
```

---

## Methodology

### 1. Data Preprocessing & Engineering
- **Medical Filtering:** Removed biologically impossible blood pressure values (systolic/diastolic) to ensure data integrity and domain accuracy.
- **Feature Engineering:** Calculated **Body Mass Index (BMI)** using height and weight to capture obesity-related risks, a critical factor in cardiology.
- **Conversion:** Converted age from days to years to facilitate intuitive Exploratory Data Analysis (EDA).

### 2. Feature Selection & Dimensionality Reduction
- **Chi-Square Selection:** Utilized statistical testing to mathematically rank and select the most significant predictors of cardiovascular disease, reducing noise.
- **PCA (Principal Component Analysis):** Analyzed variance retention across dimensions to understand dataset complexity.

### 3. Modeling & Optimization
- **Algorithm:** **XGBoost Classifier** (State-of-the-Art algorithm for tabular medical data).
- **Hyperparameter Tuning:** Conducted a comprehensive search over learning rates, max depths, and tree estimators using `RandomizedSearchCV`.
- **Validation:** Ensured model robustness and mitigated overfitting by implementing strict **5-Fold Cross Validation**.

### 4. Explainable AI (XAI)
- Integrated **SHAP** values to break the "black box" nature of tree-based models. This provides transparent visual evidence of which medical factors (e.g., Blood Pressure, Age, BMI) predominantly influenced the algorithm's prediction for any individual patient.

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kebabresing/cardiovascular-risk-prediction.git
   cd cardiovascular-risk-prediction
   ```

2. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   ```bash
   jupyter notebook Cardiovascular_Disease_Prediction.ipynb
   ```

> **Note:** Ensure you place your `cardio_train.csv` dataset file inside the `dataset/` directory before running the notebook. The script will automatically generate the `model/` directory upon execution to serialize the trained XGBoost model and standard scaler.

---

## License
This project is for educational and portfolio demonstration purposes. Feel free to use and modify the methodology.