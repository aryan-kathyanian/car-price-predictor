from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('LinearRegressionModel.pkl')
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')
    return render_template('index.html',
                           companies=companies,
                           years=year,
                           fuel_types=fuel_type)

@app.route('/get_car_models', methods=['POST'])
@cross_origin()
def get_car_models():
    data = request.get_json()
    company = data['company']
    models = sorted(car[car['company'] == company]['name'].unique().tolist())
    return jsonify({'models': models})

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))

    prediction = model.predict(input_df)
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
