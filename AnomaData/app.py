import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the saved model
with open('models/best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return "Anoma Data Model API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get data from the POST request as JSON

        if isinstance(data, dict):  # If a single row of data is passed
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)  # If multiple rows are passed
        
        # Preprocess data if necessary (like scaling, feature engineering)
        predictions = model.predict(df)
        
        return jsonify(predictions.tolist())
    
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
