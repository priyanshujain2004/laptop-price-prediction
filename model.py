# model.py
import pandas as pd
import numpy as np
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

def train_model(data):
    # Features and target
    X = data.drop(columns=['Price', 'Unnamed: 0', 'ScreenResolution', 'Memory'])
    y = data['Price']

    # Apply log transformation to stabilize target variance
    y = np.log1p(y)

    # One-hot encoding for categorical features
    categorical_features = ['Company', 'TypeName', 'Cpu', 'Gpu', 'OpSys']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), ['Inches', 'Weight', 'Ram', 'Width', 'Height', 'Has_SSD', 'Has_HDD']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Create pipeline with a RandomForestRegressor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42))
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_cv_mse = -scores.mean()
    print(f"Cross-validated Mean Squared Error: {mean_cv_mse:.2f}")

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error on Test Set: {mse:.2f}")

    return model