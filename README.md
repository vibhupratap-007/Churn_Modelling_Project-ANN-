# 🔍 Customer Churn & Salary Prediction using Deep Learning

This repository contains two end-to-end machine learning projects:

1. 🧠 **Customer Churn Prediction** using an Artificial Neural Network (ANN)
2. 📈 **Salary Prediction** using a Regression model

Both projects include data preprocessing, model training, evaluation, and deployment using **Streamlit**.

---

## 📂 Project Structure

📦 root/

- Churn_Modelling.csv # Dataset for churn prediction

- model.h5 # Trained churn classification model

- regression_model.h5 # Trained salary regression model

- scaler.pkl # Scaler used for both models

- label_encoder_gender.pkl # Label encoder for gender

- onehot_encoder_geo.pkl # One-hot encoder for geography

- experiments.ipynb # Churn model development notebook

- hyperparametertuningann.ipynb # ANN hyperparameter tuning

- prediction.ipynb # Inference testing for churn model

- salaryregression.ipynb # Salary regression notebook

- app.py # Streamlit app for churn prediction

- streamlit_regression.py # Streamlit app for salary prediction

- requirements.txt # Required Python packages

---

## 🧠 Project 1: Customer Churn Prediction

### 📌 Objective
Predict whether a bank customer will **churn (exit)** using demographic and account activity data.

### 🔧 Model
- Type: ANN (Artificial Neural Network)
- Layers: Dense (ReLU) + Dropout + Output (Sigmoid)
- Framework: TensorFlow / Keras

### 🗂 Features Used
- Credit score, Geography, Gender, Age, Tenure, Balance, etc.

### 📈 Metrics
- Accuracy, Precision, Recall, F1-Score

### 🚀 Usage
```bash
streamlit run app.py
📊 Project 2: Salary Prediction
📌 Objective
Predict the salary of an individual based on features such as experience and domain.

🔧 Model
Type: Regression (Feed-forward neural network)

Framework: TensorFlow / Keras

Install Requirements

bash
Copy
Edit
pip install -r requirements.txt
Run Streamlit Apps

For Churn Prediction:

bash
Copy
Edit
streamlit run app.py
For Salary Prediction:

bash
Copy
Edit
streamlit run streamlit_regression.py

👨‍💻 Author
Vibhu Pratap
GitHub: @yourusername

