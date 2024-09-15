# Fraud Detection Code with EDA, Model Building, and API Deployment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from flask import Flask, request, jsonify

# Step 1: Exploratory Data Analysis (EDA)

def perform_eda(data):
    # Basic info
    print(data.info())
    print(data.describe())

    # Checking class imbalance
    print('Class distribution:')
    print(data['is_fraud'].value_counts())

    # Visualizing the fraud distribution
    sns.countplot(x='is_fraud', data=data)
    plt.title('Fraud vs Legitimate Transaction Distribution')
    plt.show()

    # Visualizing transaction amount distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['transactionAmount'], bins=50)
    plt.title('Transaction Amount Distribution')
    plt.show()

    # Time-based fraud patterns (hourly)
    data['transactionDate'] = pd.to_datetime(data['transactionDate'])
    data['hour'] = data['transactionDate'].dt.hour
    fraud_hourly = data.groupby(['hour', 'is_fraud']).size().unstack().fillna(0)
    fraud_hourly.plot(kind='bar', stacked=True)
    plt.title('Fraud vs Legitimate by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Transactions')
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Load the dataset and perform EDA
fraud_train = pd.read_csv('fraudTrain.csv')
perform_eda(fraud_train)

# Step 2: Data Preprocessing

def preprocess_data(data):
    # Handle missing values
    data = data.fillna(0)

    # Feature Engineering: Time-based features from transaction timestamps
    data['transactionDate'] = pd.to_datetime(data['transactionDate'])
    data['hour'] = data['transactionDate'].dt.hour
    data['day'] = data['transactionDate'].dt.day
    data['month'] = data['transactionDate'].dt.month

    # Drop irrelevant columns
    drop_cols = ['transactionID', 'transactionDate', 'customerID', 'merchantID']
    data = data.drop(columns=drop_cols, errors='ignore')

    # One-hot encoding for categorical features (if needed)
    data = pd.get_dummies(data, drop_first=True)

    return data

# Preprocess train data
train_data = preprocess_data(fraud_train)

# Splitting into features and target
X = train_data.drop('is_fraud', axis=1)
y = train_data['is_fraud']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Building & Tuning

# Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Step 4: Model Evaluation & Comparison

def evaluate_model(y_true, y_pred, model_name):
    print(f'\n{model_name} Performance:')
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print(f'Precision: {precision_score(y_true, y_pred)}')
    print(f'Recall: {recall_score(y_true, y_pred)}')
    print(f'F1 Score: {f1_score(y_true, y_pred)}')
    print(f'ROC AUC Score: {roc_auc_score(y_true, y_pred)}')

# Evaluating models
evaluate_model(y_test, y_pred_log, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Step 5: Model Deployment as an API using Flask

app = Flask(__name__)

# Save the XGBoost model to be used in the API
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

# Load the trained model
with open('fraud_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# API Endpoint to predict fraud
@app.route('/predict', methods=['POST'])
def predict_fraud():
    data = request.json
    data_df = pd.DataFrame(data, index=[0])  # Convert to DataFrame
    data_scaled = scaler.transform(data_df)  # Scale the input

    prediction = model.predict(data_scaled)  # Predict fraud (0: legitimate, 1: fraud)
    return jsonify({'fraud_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
