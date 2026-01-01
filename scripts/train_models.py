import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def build_pipeline():
    # 1. Data Loading (Using a sample dataset for demonstration)
    # In a real scenario, you'd use: df = pd.read_csv('your_data.csv')
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    print("[v0] Data loaded successfully.")

    # 2. Data Cleaning & Preprocessing
    # Handle missing values, encoding, etc.
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Training - Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    print(f"[v0] Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds):.4f}")

    # 4. Model Training - Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_scaled, y_train)
    dt_preds = dt_model.predict(X_test_scaled)
    print(f"[v0] Decision Tree Accuracy: {accuracy_score(y_test, dt_preds):.4f}")

    # 5. Exporting Models and Scaler
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(lr_model, 'models/logistic_regression.joblib')
    joblib.dump(dt_model, 'models/decision_tree.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("[v0] Models and scaler exported to /models directory.")

if __name__ == "__main__":
    build_pipeline()
