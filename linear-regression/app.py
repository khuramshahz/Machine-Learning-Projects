from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and data
model = pickle.load(open('model.pkl', 'rb'))
car = pd.read_csv('Cleaned_data.csv')

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', 
                           companies=companies, 
                           car_models=car_models, 
                           years=years,
                           fuel_types=fuel_types)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    # Predicting
    input_data = pd.DataFrame([[car_model, company, int(year), int(driven), fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    prediction = model.predict(input_data)[0]

    output = f"Estimated Price: {np.round(prediction, 2)} RS"
    
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')

    return render_template('index.html', 
                           companies=companies, 
                           car_models=car_models, 
                           years=years, 
                           fuel_types=fuel_types,
                           prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
