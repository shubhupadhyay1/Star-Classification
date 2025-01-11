import joblib
import pandas as pd

def load_model(model_path='models/star_classifier.pkl'):
    """Loads a saved model from disk."""
    model = joblib.load(model_path)
    return model

def predict_new_data(model, new_data):
    """Runs prediction on new data."""
    predictions = model.predict(new_data)
    return predictions
