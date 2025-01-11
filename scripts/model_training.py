import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.data_processing import load_and_clean_data, feature_engineering


def train_and_save_model(df, output_path='models/star_classifier.pkl'):
    """Trains a Random Forest model and saves it to disk."""
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    
    joblib.dump(model, output_path)
    print(f"Model saved at {output_path}")

if __name__ == '__main__':
    print("Columns in DataFrame:", df.columns)
    print(df.head())

    df = load_and_clean_data('data/raw/Star99999_raw.csv')
    df = feature_engineering(df)
    train_and_save_model(df)
