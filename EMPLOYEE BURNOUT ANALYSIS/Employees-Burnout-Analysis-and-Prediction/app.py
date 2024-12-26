from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('models/linear_regression.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the form
    designation_type = request.form['designation_type']
    resource_allocation = float(request.form['resource_allocation'])
    mental_fatigue_score = float(request.form['mental_fatigue_score'])
    company_type = request.form['company_type']
    wfh_setup_available = request.form['wfh_setup_available']
    gender = request.form['gender']
    
    # One-hot encode categorical features
    designation_type_manager = 1 if designation_type == 'Manager' else 0
    company_type_service = 1 if company_type == 'Service' else 0
    wfh_setup_available_yes = 1 if wfh_setup_available == 'Yes' else 0
    gender_male = 1 if gender == 'Male' else 0

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Designation Type_Manager': [designation_type_manager],
        'Resource Allocation': [resource_allocation],
        'Mental Fatigue Score': [mental_fatigue_score],
        'Company Type_Service': [company_type_service],
        'WFH Setup Available_Yes': [wfh_setup_available_yes],
        'Gender_Male': [gender_male]
    })

    # Add any missing columns that the scaler expects
    missing_cols = set(scaler.feature_names_in_) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    # Align the input data columns with the model's expected columns
    input_data = input_data[scaler.feature_names_in_]

    # Perform standard scaling on the input data
    scaled_data = scaler.transform(input_data)

    # Make a prediction using the model
    prediction = model.predict(scaled_data)[0]
    rounded_prediction = round(prediction, 2)

    return render_template('index.html', prediction=rounded_prediction)

if __name__ == "__main__":
    app.run(debug=True)
