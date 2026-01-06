# Customer Churn Prediction – End-to-End ML Pipeline

## Overview
Customer churn refers to customers who stop using a company’s service. Predicting churn helps businesses take proactive actions to retain high-risk customers.

This project implements a complete end-to-end machine learning pipeline to predict customer churn, covering exploratory data analysis, preprocessing, modeling, evaluation, and final model selection.

## Dataset
- Name: Telco Customer Churn Dataset  
- Target Variable: Churn (Yes / No)  
- Objective: Predict whether a customer will churn based on customer profile, service usage, and billing information.

## Project Structure
Note Books/
- 01_eda.ipynb – Exploratory Data Analysis  
- 02_preprocessing.ipynb – Data cleaning and feature engineering  
- 03_modeling.ipynb – Modeling, evaluation, and model selection  

Data/
- raw/ – Original dataset  

Models/
- final_churn_model.pkl – Saved trained model  

## Workflow
1. Exploratory Data Analysis  
   - Analyzed churn distribution and class imbalance  
   - Studied relationships between churn, tenure, charges, and contracts  
   - Identified business-relevant churn drivers  

2. Data Preprocessing  
   - Data type corrections and missing value handling  
   - Feature scaling and categorical encoding using pipelines  
   - Reproducible preprocessing with ColumnTransformer  

3. Modeling and Evaluation  
   - Baseline Logistic Regression  
   - Class-weighted Logistic Regression to handle class imbalance  
   - Decision threshold optimization to improve recall  
   - Cross-validation to assess model robustness  

## Final Model
Class-weighted Logistic Regression was selected as the final model because it provides improved recall for churn customers, good interpretability, and stable performance across validation folds. The model prioritizes identifying churn-prone customers to support retention strategies.

## Key Results
- Improved recall for churn customers after imbalance handling  
- Business-aware decision threshold tuning  
- Clean and reusable machine learning pipeline  

## Tech Stack
- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  
- Git and GitHub  

## How to Run
conda create -n churn-env python=3.10  
conda activate churn-env  
pip install -r requirements.txt  
jupyter notebook  

## Next Steps
- Deploy prediction API using FastAPI  
- Add MLflow experiment tracking  
- Containerize the pipeline using Docker  
