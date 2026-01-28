# Customer Churn Prediction – End-to-End ML Pipeline

## Overview
Customer churn means customers are no longer using a company’s service. Knowing who might churn allows businesses to take proactive measures to retain these high-risk customers.

This project implements a complete end-to-end machine learning pipeline for customer churn prediction, covering data exploration, preprocessing, model training, evaluation, experiment tracking, and API deployment.

## Dataset
- Name: Telco Customer Churn Dataset  
- Target Variable: Churn (Yes / No)  
- Objective: Predict whether a customer will churn based on customer profile, service usage, and billing information.

## Project Structure
 Note Books
- 01_eda.ipynb – Exploratory Data Analysis  
- 02_preprocessing.ipynb – Data cleaning and feature engineering  
- 03_modeling.ipynb – Modeling, evaluation, and model selection  

Data
- raw – Original dataset  

Models
- final_churn_model.pkl – Saved trained model  

api
- main.py
- predict.py
- schemas.py

requirements.txt

README.md

## Workflow
1. Exploratory Data Analysis  
   - Analyzed churn distribution and identified class imbalance  
   - Explored relationships between churn, tenure, monthly charges, contract type, and customer demographics  
   - Extracted business-relevant insights affecting churn behavior

2. Data Preprocessing  
   - Data type corrections and missing value handling  
   - Scaled numerical features and encoded categorical variables  
   - Built a reproducible preprocessing pipeline using ColumnTransformer 

3. Modeling and Evaluation  
   - Trained a baseline Logistic Regression model  
   - Class-weighted Logistic Regression to handle class imbalance  
   - Decision threshold optimization to improve recall  
   - Trained a Random Forest model for comparison

## Experiment Tracking
Model experiments were tracked using MLflow:
   - Logged model parameters, metrics, and artifacts
   - Compared baseline, weighted, and tree-based models
   - Used MLflow UI to support data-driven model selection

## Final Model
Class-weighted Logistic Regression was selected as the final model because it provides improved recall for churn customers, good interpretability. The model prioritizes identifying churn-prone customers to support retention strategies.

## API Deployment
A prediction API was built using FastAPI:
   - Accepts customer data via REST requests
   - Applies the trained model and preprocessing pipeline
   - Returns churn probability and prediction outcome
   - Includes interactive documentation via Swagger UI

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
- MLflow
- FastAPI

## How to Run
conda create -n churn-env python=3.10  
conda activate churn-env  
pip install -r requirements.txt  
jupyter notebook  

## To run the API
uvicorn api.main:app --reload

