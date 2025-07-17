from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and label encoder
with open('flight_price_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    le = saved_data['label_encoder']

# Get unique airlines from the label encoder
airlines = le.classes_

@app.route('/')
def index():
    return render_template('index.html', airlines=airlines)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        airline = request.form['airline']
        duration = float(request.form['duration'])
        
        # Encode airline
        airline_encoded = le.transform([airline])[0]
        
        # Prepare input for model
        input_data = np.array([[airline_encoded, duration]])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)