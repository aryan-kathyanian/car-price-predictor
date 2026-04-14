# retrain_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


df = pd.read_csv('Cleaned_Car_data.csv')

df = df.dropna()
df['year'] = df['year'].astype(int)
df['kms_driven'] = df['kms_driven'].astype(int)


X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']


categorical_features = ['name', 'company', 'fuel_type']


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')


model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

joblib.dump(model_pipeline, 'LinearRegressionModel.pkl')
print(" Model retrained and saved as LinearRegressionModel.pkl")
