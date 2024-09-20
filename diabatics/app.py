from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Check and print the current working directory
print(f"Current working directory: {os.getcwd()}")

# Load the trained model and scaler
model_path = 'logistic_regression_model.pkl'  # Update this path as needed
scaler_path = 'scaler.pkl'  # Update this path as needed

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    model, scaler = None, None

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('result.html', result_message="Model or scaler not loaded. Check the server logs.")

    try:
        # Get form data and ensure all fields are present
        required_fields = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        input_data = []

        for field in required_fields:
            value = request.form.get(field)
            if not value:
                return render_template('result.html', result_message=f"Error: Missing field {field}")
            try:
                input_data.append(float(value))
            except ValueError:
                return render_template('result.html', result_message=f"Error: Invalid value for {field}")
        
        # Convert the input data to numpy array and reshape it for scaling
        input_data = np.array(input_data).reshape(1, -1)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_data_scaled)[0]

        # For simplicity, assume that if the prediction is above 0.5, the user has diabetes
        result_message = "The model predicts you have DIABETES." if prediction > 0.5 else "The model predicts you do NOT HAVE DIABETES."
    
    except Exception as e:
        result_message = f"Error: {e}"

    return render_template('result.html', result_message=result_message)

if __name__ == "__main__":
    app.run(debug=True)
