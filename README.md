# Car Price Predictor

A Flask-based machine learning web app that predicts used car prices from user inputs such as company, model, year, fuel type, and kilometers driven.

## Features

- Predicts car price using a trained `LinearRegression` pipeline.
- Dynamically loads available car models based on selected company.
- Simple browser UI for entering car details and viewing estimated price.
- Includes a retraining script to regenerate the model from cleaned data.

## Project Structure

- `application.py` - Flask app with routes for UI, model lookup, and prediction.
- `retrain_model.py` - Retrains and saves `LinearRegressionModel.pkl`.
- `templates/index.html` - Frontend form and JavaScript for API calls.
- `static/css/style.css` - UI styling.
- `Cleaned_Car_data.csv` - Clean dataset used for training and app options.
- `quikr_car.csv` - Raw/source car dataset.
- `Procfile` - Production start command (`gunicorn application:app`).

## Requirements

Install Python packages:

- `flask`
- `flask-cors`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `gunicorn` (for deployment)

## Setup

1. Clone or download this project.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
pip install flask flask-cors pandas numpy scikit-learn joblib gunicorn
```

## Run the App (Local)

From the project root:

```bash
python application.py
```

Then open:

- `http://127.0.0.1:5000/`

## Retrain the Model

If you update `Cleaned_Car_data.csv`, retrain the model:

```bash
python retrain_model.py
```

This recreates `LinearRegressionModel.pkl`, which is loaded by `application.py`.

## API Endpoints

- `GET /` - Serves the web interface.
- `POST /get_car_models` - Returns models for selected company (JSON).
- `POST /predict` - Returns predicted price as text response.

```

## Notes

- Make sure `LinearRegressionModel.pkl` and `Cleaned_Car_data.csv` are present in the project root before running the app.
- Prediction quality depends on the training data quality and model choice.
