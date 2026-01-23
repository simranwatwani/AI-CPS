import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import warnings
import shutil
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

try:
    from web_scrapper import load_data_from_github
    WEB_SCRAPER_AVAILABLE = True
except ImportError:
    WEB_SCRAPER_AVAILABLE = False
    print("Web scraper module not found. Using local files only.")

warnings.filterwarnings('ignore')
np.random.seed(42)

def get_root_directory():
    current_dir = Path.cwd()
    
    if current_dir.name == 'code':
        root_dir = current_dir.parent
        print(f"Found 'code' folder, setting root to: {root_dir}")
    else:
        root_dir = current_dir
        print(f"Root directory: {root_dir}")
    
    return root_dir

def create_directories(root_dir):
    dirs_to_create = [
        root_dir / 'output',
        root_dir / 'images' / 'learningBase_german_car' / 'train',
        root_dir / 'images' / 'learningBase_german_car' / 'validation',
        root_dir / 'images' / 'activationBase_german_car',
        root_dir / 'images' / 'knowledgeBase_german_car',
        root_dir / 'images' / 'codeBase_german_car',
        root_dir / 'scenarios' / 'apply_annSolution_german_car' / 'x86_64',
        root_dir / 'scenarios' / 'apply_olsSolution_german_car' / 'x86_64'
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory.relative_to(root_dir)}")
    
    return dirs_to_create

def load_data(root_dir):
    if WEB_SCRAPER_AVAILABLE:
        try:
            print("Attempting to load data from Kaggle...")
            df = load_data_from_github()
            print(f"Kaggle data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return df
        except Exception as kaggle_error:
            print(f"Kaggle loading failed: {kaggle_error}")
            print("Trying local files as backup...")
    
    data_paths_to_try = [
        root_dir / 'kaggle_data' / 'autos.csv',
        root_dir / 'data' / 'autos.csv',
        root_dir / 'autos.csv'
    ]
    
    for data_path in data_paths_to_try:
        if data_path.exists():
            print(f"Found data at: {data_path}")
            try:
                df = pd.read_csv(data_path, encoding='latin1')
                print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
                return df
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
                continue
    
    print("No data file found. Creating sample data for demonstration.")
    np.random.seed(42)
    n_samples = 1000
    data = {
        'price': np.random.randint(500, 50000, n_samples),
        'yearOfRegistration': np.random.randint(1990, 2016, n_samples),
        'powerPS': np.random.randint(40, 500, n_samples),
        'kilometer': np.random.randint(0, 300000, n_samples),
        'vehicleType': np.random.choice(['limousine', 'kombi', 'suv', 'kleinwagen'], n_samples),
        'fuelType': np.random.choice(['benzin', 'diesel', 'elektro'], n_samples),
        'gearbox': np.random.choice(['manuell', 'automatik'], n_samples),
        'brand': np.random.choice(['volkswagen', 'bmw', 'mercedes_benz', 'audi'], n_samples)
    }
    df = pd.DataFrame(data)
    print(f"Created demonstration dataset: {df.shape[0]:,} rows")
    return df

def clean_data(df):
    initial_count = len(df)
    
    df_clean = df[
        (df['price'] >= 500) & (df['price'] <= 50000) &
        (df['yearOfRegistration'] >= 1950) & (df['yearOfRegistration'] <= 2016) &
        (df['powerPS'] >= 40) & (df['powerPS'] <= 500) &
        (df['kilometer'] <= 300000)
    ].copy()
    
    cleaned_count = len(df_clean)
    cleaning_percentage = (cleaned_count / initial_count) * 100
    
    print(f"Data cleaning completed:")
    print(f"Initial samples: {initial_count:,}")
    print(f"After cleaning: {cleaned_count:,} ({cleaning_percentage:.1f}% retained)")
    
    return df_clean, initial_count, cleaned_count, cleaning_percentage

def engineer_features(df):
    df['age'] = 2016 - df['yearOfRegistration']
    df['age'] = df['age'].clip(0, 30)
    df['log_price'] = np.log1p(df['price'])
    
    categorical_cols = ['vehicleType', 'fuelType', 'gearbox', 'brand']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    encoded_features = []
    numerical_features = ['age', 'powerPS', 'kilometer']
    encoded_features.extend(numerical_features)
    
    for col in categorical_cols:
        if col in df.columns:
            top_cats = df[col].value_counts().nlargest(5).index
            for cat in top_cats:
                new_col = f"{col}_{cat}"
                df[new_col] = (df[col] == cat).astype(int)
                encoded_features.append(new_col)
            df[f"{col}_other"] = (~df[col].isin(top_cats)).astype(int)
            encoded_features.append(f"{col}_other")
    
    X = df[encoded_features]
    y = df['log_price']
    y_original = df['price']
    
    print(f"Created {len(encoded_features)} features for modeling")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features (encoded): {len(encoded_features) - len(numerical_features)}")
    
    return X, y, y_original, encoded_features

def split_and_scale_data(X, y, y_original):
    X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
        X, y, y_original, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data split completed:")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features scaled: {X.shape[1]}")
    
    return (X_train, X_test, y_train, y_test, y_train_orig, y_test_orig, 
            X_train_scaled, X_test_scaled, scaler)

def save_csv_files(X, y_original, X_train, y_train_orig, X_test, y_test_orig, encoded_features, root_dir):
    print("Creating CSV files with correct formats...")
    
    output_dir = root_dir / 'output'
    
    print("1. Creating 'joint_data_collection.csv' (features + price)...")
    joint_data = pd.concat([X, y_original], axis=1)
    joint_data.to_csv(output_dir / 'joint_data_collection.csv', index=False)
    print(f"Saved: {len(joint_data):,} rows, {joint_data.shape[1]} columns")
    
    print("2. Creating 'training_data.csv' (features + price)...")
    training_df = pd.DataFrame(X_train, columns=encoded_features)
    training_df['price'] = y_train_orig
    training_df.to_csv(output_dir / 'training_data.csv', index=False)
    print(f"Saved: {len(training_df):,} rows, {training_df.shape[1]} columns")
    print(f"Price column: {'price' in training_df.columns}")
    
    print("3. Creating 'test_data.csv' (features + price)...")
    test_df = pd.DataFrame(X_test, columns=encoded_features)
    test_df['price'] = y_test_orig
    test_df.to_csv(output_dir / 'test_data.csv', index=False)
    print(f"Saved: {len(test_df):,} rows, {test_df.shape[1]} columns")
    print(f"Price column: {'price' in test_df.columns}")
    
    print("4. Creating 'activation_data.csv' (FEATURES ONLY - NO PRICE)...")
    activation_sample = pd.DataFrame(X_test.iloc[[0]], columns=encoded_features)
    activation_sample.to_csv(output_dir / 'activation_data.csv', index=False)
    print(f"Saved: 1 sample, {activation_sample.shape[1]} features")
    print(f"Price column: {'price' in activation_sample.columns} (should be False)")
    
    return training_df, test_df, activation_sample

def train_ols_model(X_train_scaled, y_train, X_test_scaled, y_test_orig, root_dir):
    print("Training Ordinary Least Squares (OLS) model...")
    
    X_train_ols = sm.add_constant(X_train_scaled)
    X_test_ols = sm.add_constant(X_test_scaled)
    
    ols_model = sm.OLS(y_train, X_train_ols).fit()
    
    y_pred_ols_log = ols_model.predict(X_test_ols)
    y_pred_ols = np.expm1(y_pred_ols_log)
    
    ols_mae = mean_absolute_error(y_test_orig, y_pred_ols)
    ols_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_ols))
    ols_r2 = r2_score(y_test_orig, y_pred_ols)
    
    model_path = root_dir / 'output' / 'currentOlsSolution.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(ols_model, f)
    
    print(f"OLS Model Performance:")
    print(f"MAE: €{ols_mae:,.2f}")
    print(f"RMSE: €{ols_rmse:,.2f}")
    print(f"R²: {ols_r2:.4f}")
    print(f"Model saved: {model_path.relative_to(root_dir)}")
    
    return ols_model, ols_mae, ols_rmse, ols_r2, y_pred_ols

