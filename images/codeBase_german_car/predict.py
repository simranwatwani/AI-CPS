#!/usr/bin/env python3
"""
German Used Car Price Prediction - Fixed Version
University of Potsdam - AI-CPS Project
"""

import pandas as pd
import numpy as np
import pickle
import datetime

# ── Use the same paths as in your Docker setup ────────────────────────────────
MODEL_PATH = "/tmp/knowledgeBase/currentAiSolution.pkl"
SCALER_PATH = "/tmp/knowledgeBase/scaler.pkl"
TRAINING_PATH = "/tmp/learningBase/train/training_data.csv"   # contains price
ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"   # no price

print("=" * 70)
print("🚗 GERMAN CAR AI - PRICE PREDICTION SYSTEM")
print("University of Potsdam - AI-CPS Project")
print("=" * 70)
print(f"Timestamp: {datetime.datetime.now()}")
print("✅ THIS IS THE NEW UPDATED CODE!")
print("=" * 70)

try:
    # Load your ANN model and scaler
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✓ ANN model loaded")
    
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
    
    # Load training data (to find similar cars + compute median/interval)
    training_data = pd.read_csv(TRAINING_PATH)  # last column is 'price'
    print(f"✓ Training data loaded: {len(training_data)} cars")
    
    # Load activation input (the car we want to predict)
    activation = pd.read_csv(ACTIVATION_PATH)   # shape should be (1, n_features)
    print(f"✓ Activation data loaded: {len(activation)} car(s), {len(activation.columns)} features")
    
    # Scale the input features (your model was trained on scaled data)
    activation_scaled = scaler.transform(activation)
    
    # 1. Predict using ANN model (log-price → real price)
    predicted_log = model.predict(activation_scaled)[0]
    predicted_price = np.expm1(predicted_log)  # reverse log1p transformation
    
    # 2. Find similar cars in training data
    numeric_cols = ['age', 'powerPS', 'kilometer']
    categorical_cols = [c for c in activation.columns if c not in numeric_cols]
    
    similar_cars = training_data.copy()
    
    # Exact match on categorical (one-hot) columns
    for col in categorical_cols:
        if col in similar_cars.columns:
            similar_cars = similar_cars[similar_cars[col] == activation[col].values[0]]
    
    # ±10% range on numeric features
    for col in numeric_cols:
        if col in similar_cars.columns:
            val = activation[col].values[0]
            similar_cars = similar_cars[
                (similar_cars[col] >= 0.9 * val) & (similar_cars[col] <= 1.1 * val)
            ]
    
    # 3. Compute median and realistic interval
    if not similar_cars.empty:
        median_price = similar_cars['price'].median()
        min_price = similar_cars['price'].min()
        max_price = similar_cars['price'].max()
        num_similar = len(similar_cars)
    else:
        # Fallback when no similar cars are found
        median_price = predicted_price
        min_price = predicted_price * 0.85
        max_price = predicted_price * 1.15
        num_similar = 0
    
    # ── Print nice results ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("💰 PREDICTION RESULTS:")
    print("-" * 40)
    print(f"Expected Price (ANN model):       €{predicted_price:,.2f}")
    print(f"Median Price (similar cars):      €{median_price:,.2f}")
    print(f"Realistic Price Interval:         €{min_price:,.2f} – €{max_price:,.2f}")
    print(f"Number of similar cars found:     {num_similar}")
    
    # Show car details
    print("\n🚗 CAR DETAILS:")
    print("-" * 40)
    if 'age' in activation.columns:
        print(f"Age: {int(activation['age'].iloc[0])} years")
    if 'powerPS' in activation.columns:
        print(f"Power: {int(activation['powerPS'].iloc[0])} PS")
    if 'kilometer' in activation.columns:
        print(f"Mileage: {int(activation['kilometer'].iloc[0]):,} km")
    
    # Find vehicle type
    vehicle_cols = [col for col in activation.columns if col.startswith('vehicleType_')]
    for col in vehicle_cols:
        if activation[col].iloc[0] == 1:
            print(f"Type: {col.replace('vehicleType_', '').capitalize()}")
            break
    
    # Find brand
    brand_cols = [col for col in activation.columns if col.startswith('brand_')]
    for col in brand_cols:
        if activation[col].iloc[0] == 1:
            print(f"Brand: {col.replace('brand_', '').upper()}")
            break
    
    print("\n" + "=" * 70)
    print("✅ PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

except FileNotFoundError as e:
    print(f"\n❌ FILE NOT FOUND ERROR: {e}")
    print("Check if these files exist in the container:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {TRAINING_PATH}")
    print(f"  - {ACTIVATION_PATH}")
    print("\nRun this to check: docker run --rm german-car-code-v2 ls -la /tmp/")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
