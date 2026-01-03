# Customer Churn Prediction with Automated ML Pipeline

## Project Overview
This project predicts customer churn using supervised machine learning.
It follows a full end-to-end workflow including EDA, preprocessing,
modeling, evaluation, and model selection.

## Dataset
Telco Customer Churn dataset.

## Project Structure
- notebooks/ → EDA, preprocessing, modeling notebooks
- data/ → raw and processed datasets
- models/ → trained and saved ML models

## Approach
- Exploratory Data Analysis (EDA)
- Data preprocessing using pipelines
- Baseline Logistic Regression
- Class imbalance handling
- Threshold optimization
- Cross-validation
- Final model selection

## Model Selection
Logistic Regression with class weighting was selected as the final model
due to its strong recall performance for churn customers and interpretability.

## Key Results
- Improved recall for churn customers
- Business-aware threshold tuning
- Robust preprocessing pipeline

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Next Steps
- API deployment using FastAPI