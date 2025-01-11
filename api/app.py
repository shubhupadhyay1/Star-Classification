from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('../models/star_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predicting star types."""
    data = request.get_json()  # Expecting JSON input
    df = pd.DataFrame(data)  # Convert JSON to DataFrame
    predictions = model.predict(df)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
