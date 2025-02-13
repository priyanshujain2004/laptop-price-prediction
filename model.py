# model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re

def load_data():
    # Load the data
    file_path = "laptop_data.csv"
    data = pd.read_csv(file_path)
    data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)
    data['Ram'] = data['Ram'].str.replace('GB', '').astype(int)

    # Feature engineering: extract screen resolution width and height
    data['Width'] = data['ScreenResolution'].apply(lambda x: int(re.search(r'(\d+)x', x).group(1)) if re.search(r'(\d+)x', x) else 1366)
    data['Height'] = data['ScreenResolution'].apply(lambda x: int(re.search(r'x(\d+)', x).group(1)) if re.search(r'x(\d+)', x) else 768)

    # Feature engineering: split Memory into SSD and HDD sizes
    data['Has_SSD'] = data['Memory'].apply(lambda x: 1 if 'SSD' in x else 0)
    data['Has_HDD'] = data['Memory'].apply(lambda x: 1 if 'HDD' in x else 0)
    
    # Handling outliers: Cap prices above a threshold (e.g., 500,000)
    data['Price'] = np.where(data['Price'] > 500000, 500000, data['Price'])

    return data

def train_and_save_model(data, model_path='model.pkl'):
    # Features and target
    X = data.drop(columns=['Price', 'Unnamed: 0', 'ScreenResolution', 'Memory'])
    y = data['Price']
    y = np.log1p(y)  # Apply log transformation

    # One-hot encoding for categorical features
    categorical_features = ['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), ['Inches', 'Weight', 'Ram', 'Width', 'Height', 'Has_SSD', 'Has_HDD']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42))
    ])

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model

def load_model(model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
