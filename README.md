## Heart Disease Risk Prediction using Logistic Regression
Tomas Felipe Ramirez Alvarez

### Project Overview

This project implements a logistic regression model to predict the presence of heart disease based on clinical features such as age, cholesterol, blood pressure, and exercise-related metrics. Logistic regression is a classic method for binary classification and is suitable for early identification of patients at risk.

The goal is to explore end-to-end predictive modeling: from data loading, preprocessing, training, visualization, and evaluation, to deployment in a cloud environment (Amazon SageMaker).

Dataset
    - Source: Kaggle - Heart Disease Dataset

Dataset Details:
    - Number of patients: 303

Features: 14 clinical attributes, including:

    - Age: Age of patient (29–77)
    - Sex: Male/Female (1/0)
    - Chest pain type
    - BP: Resting blood pressure
    - Cholesterol (mg/dL)
    - FBS over 120
    - EKG results
    - Max HR
    - Exercise angina
    - ST depression
    - Slope of ST
    - Number of vessels fluro
    - Thallium
    - Target: Heart Disease (Presence = 1, Absence = 0)
    - Disease prevalence: ~55%