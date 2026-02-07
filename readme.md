# Author

NAKSHATRA MEENA

nakshatra301407@gmail.com

* Project:

# Customer Product Prediction

A Machine Learning–based web application that predicts the most likely product category a customer will purchase based on demographic and behavioral data.  
The system uses a trained ML model and is deployed using Flask with an interactive web interface.

---

## Project Overview

Customer Product Prediction is a "supervised machine learning classification project" designed to assist businesses in understanding customer behavior and improving product recommendations.

This project was developed as a "Final-Year Engineering Project" and demonstrates the complete ML lifecycle:
- Data preprocessing
- Model training and evaluation
- Model serialization
- Web deployment using Flask

---

## Problem Statement

To predict the "next product category" a customer is likely to purchase using historical customer data such as age, income, spending behavior, browsing time, and past purchases.

---

## Machine Learning Approach

- Type: Multi-class Classification  
- Algorithm Used: Random Forest Classifier  
- Why Random Forest?
  - Handles non-linear data well
  - Reduces overfitting
  - Provides high accuracy for tabular datasets

---

## Tech Stack

- Programming Language: Python  
- Machine Learning: Scikit-learn  
- Web Framework: Flask  
- Data Processing: Pandas, NumPy  
- Frontend: HTML, CSS  
- Model Serialization: Pickle  

---

## Project Structure

Customer-Product-Prediction/
│── app.py # Flask application
│── train_model.py # Model training script
│── model.pkl # Trained ML model
│── scaler.pkl # Feature scaler
│── requirements.txt
│── README.md
│── dataset/
│ └── customer_data.csv
│── templates/
│ └── index.html
│── static/
│ └── style.css


---

## How to Run the Project

### 1️ Install Dependencies
```bash
pip install -r requirements.txt

2️ Train the Model

python train_model.py

This will generate:

    model.pkl

    scaler.pkl

3️ Run the Flask App

python app.py

4️ Open in Browser

http://127.0.0.1:5000/

 Application Features

    User-friendly web interface

    Real-time customer product prediction

    Dropdown-based inputs (no manual encoding)

    Human-readable prediction output

    Scalable ML deployment architecture

 Dataset Description

The dataset is synthetically generated and includes:

    Customer demographics (age, gender, income)

    Behavioral metrics (spending score, browsing time)

    Purchase history

    Target product category

This ensures privacy while maintaining realistic patterns for learning.
 Results & Insights

    Spending score and past purchases are strong predictors

    Random Forest provides reliable performance on customer data

    Model can be extended for real-world recommendation systems

 Future Enhancements

    Add confidence/probability scores

    Integrate XGBoost or Neural Networks

    Deploy on cloud platforms (AWS / Render)

    Connect with real-time databases

    Add user authentication

 Academic Relevance

This project fulfills final-year project requirements by covering:

    Problem definition

    System design

    Machine learning implementation

    Deployment

    Evaluation and future scope
