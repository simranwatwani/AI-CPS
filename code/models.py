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
def train_ann_model(X_train_scaled, y_train, X_test_scaled, y_test_orig, root_dir):
    print("Training Artificial Neural Network (ANN) model...")
    
    ann_model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=100,
        random_state=42,
        verbose=False
    )
    
    ann_model.fit(X_train_scaled, y_train)
    
    y_pred_ann_log = ann_model.predict(X_test_scaled)
    y_pred_ann = np.expm1(y_pred_ann_log)
    
    ann_mae = mean_absolute_error(y_test_orig, y_pred_ann)
    ann_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_ann))
    ann_r2 = r2_score(y_test_orig, y_pred_ann)
    
    model_path = root_dir / 'output' / 'currentAiSolution.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(ann_model, f)
    
    print(f"ANN Model Performance:")
    print(f"MAE: €{ann_mae:,.2f}")
    print(f"RMSE: €{ann_rmse:,.2f}")
    print(f"R²: {ann_r2:.4f}")
    print(f"Architecture: {ann_model.hidden_layer_sizes}")
    print(f"Model saved: {model_path.relative_to(root_dir)}")
    
    return ann_model, ann_mae, ann_rmse, ann_r2, y_pred_ann

def create_visualizations(y_test_orig, y_pred_ols, y_pred_ann, ols_r2, ann_r2, ols_mae, ann_mae, root_dir):
    print("Creating diagnostic visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].scatter(y_test_orig, y_pred_ols, alpha=0.3, s=10, color='blue')
        axes[0, 0].plot([y_test_orig.min(), y_test_orig.max()], 
                       [y_test_orig.min(), y_test_orig.max()], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Price (€)')
        axes[0, 0].set_ylabel('Predicted Price (€)')
        axes[0, 0].set_title(f'OLS Model: R² = {ols_r2:.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].scatter(y_test_orig, y_pred_ann, alpha=0.3, s=10, color='green')
        axes[0, 1].plot([y_test_orig.min(), y_test_orig.max()], 
                       [y_test_orig.min(), y_test_orig.max()], 'r--', linewidth=2)
        axes[0, 1].set_xlabel('Actual Price (€)')
        axes[0, 1].set_ylabel('Predicted Price (€)')
        axes[0, 1].set_title(f'ANN Model: R² = {ann_r2:.3f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        ols_errors = np.abs(y_test_orig - y_pred_ols)
        ann_errors = np.abs(y_test_orig - y_pred_ann)
        
        axes[1, 0].hist(ols_errors, bins=50, alpha=0.7, color='blue', 
                        label=f'OLS (Mean: €{ols_errors.mean():,.0f})', density=True)
        axes[1, 0].hist(ann_errors, bins=50, alpha=0.7, color='green', 
                        label=f'ANN (Mean: €{ann_errors.mean():,.0f})', density=True)
        axes[1, 0].set_xlabel('Absolute Error (€)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Error Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        models = ['OLS', 'ANN']
        mae_values = [ols_mae, ann_mae]
        r2_values = [ols_r2, ann_r2]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x - width/2, mae_values, width, label='MAE (€)', color='orange')
        bars2 = axes[1, 1].bar(x + width/2, r2_values, width, label='R²', color='purple')
        
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Performance Metric')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('German Used Car Price Prediction - University of Potsdam', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        output_path = root_dir / 'output' / 'model_diagnostics.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Diagnostic plots saved: {output_path.relative_to(root_dir)}")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        


def create_complete_comparison_plot(ols_mae, ols_rmse, ols_r2, ann_mae, ann_rmse, ann_r2, root_dir):
    try:
        print("  Creating complete metrics comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Data preparation
        metrics = ['MAE (€)', 'RMSE (€)', 'R²']
        ols_values = [ols_mae, ols_rmse, ols_r2]
        ann_values = [ann_mae, ann_rmse, ann_r2]
        
        colors = ['blue', 'green']
        
        # Plot 1: Side-by-side bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, ols_values, width, label='OLS', color=colors[0], alpha=0.7)
        bars2 = ax1.bar(x + width/2, ann_values, width, label='ANN', color=colors[1], alpha=0.7)
        
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Value')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars, values in zip([bars1, bars2], [ols_values, ann_values]):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                label = f'€{value:,.0f}' if bar.get_x() < 2 else f'{value:.4f}'
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Improvement percentages
        improvements = [
            ((ols_mae - ann_mae) / ols_mae) * 100,  # MAE improvement
            ((ols_rmse - ann_rmse) / ols_rmse) * 100,  # RMSE improvement
            (ann_r2 - ols_r2) * 100  # R² improvement
        ]
        
        colors_improvement = ['green' if imp > 0 else 'red' for imp in improvements]
        bars_imp = ax2.bar(metrics, improvements, color=colors_improvement, alpha=0.7)
        
        ax2.set_xlabel('Metric')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('ANN Improvement over OLS')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, imp in zip(bars_imp, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.1f}%', ha='center', va='bottom' if imp >= 0 else 'top',
                    fontsize=9, fontweight='bold')
        
        # Plot 3: Radar chart
        categories = ['MAE', 'RMSE', 'R²']
        
        # Normalize for radar chart
        max_mae = max(ols_mae, ann_mae)
        max_rmse = max(ols_rmse, ann_rmse)
        
        ols_normalized = [
            1 - (ols_mae / max_mae),  # Lower MAE is better
            1 - (ols_rmse / max_rmse), # Lower RMSE is better
            ols_r2  # Higher R² is better
        ]
        
        ann_normalized = [
            1 - (ann_mae / max_mae),
            1 - (ann_rmse / max_rmse),
            ann_r2
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ols_normalized += ols_normalized[:1]
        ann_normalized += ann_normalized[:1]
        categories += [categories[0]]
        
        ax3 = plt.subplot(223, polar=True)
        ax3.plot(angles, ols_normalized, 'o-', linewidth=2, label='OLS', color='blue')
        ax3.fill(angles, ols_normalized, alpha=0.25, color='blue')
        ax3.plot(angles, ann_normalized, 'o-', linewidth=2, label='ANN', color='green')
        ax3.fill(angles, ann_normalized, alpha=0.25, color='green')
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories[:-1])
        ax3.set_ylim(0, 1)
        ax3.set_title('Performance Radar Chart')
        ax3.legend(loc='upper right')
        ax3.grid(True)
        
        # Plot 4: Winner summary
        ax4.axis('off')
        
        summary_text = (
            ' PERFORMANCE WINNERS \n\n'
            f'• R² Score: {"ANN" if ann_r2 > ols_r2 else "OLS"}\n'
            f'  (ANN: {ann_r2:.4f} vs OLS: {ols_r2:.4f})\n\n'
            f'• MAE: {"ANN" if ann_mae < ols_mae else "OLS"}\n'
            f'  (ANN: €{ann_mae:,.0f} vs OLS: €{ols_mae:,.0f})\n\n'
            f'• RMSE: {"ANN" if ann_rmse < ols_rmse else "OLS"}\n'
            f'  (ANN: €{ann_rmse:,.0f} vs OLS: €{ols_rmse:,.0f})\n\n'
            f' Overall Best: {"ANN" if (ann_r2 > ols_r2 and ann_mae < ols_mae) else "OLS"}'
        )
        
        ax4.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=12, transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('Complete Model Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = root_dir / 'output' / 'complete_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Complete comparison plot saved: {output_path.relative_to(root_dir)}")
        
    except Exception as e:
        print(f"   Complete comparison plot error: {e}")
        

def copy_files_to_images(training_df, test_df, activation_sample, root_dir):
    print("Copying files to Docker image directories...")
    
    print("1. Copying to learningBase_german_car...")
    train_path = root_dir / 'images' / 'learningBase_german_car' / 'train' / 'training_data.csv'
    validation_path = root_dir / 'images' / 'learningBase_german_car' / 'validation' / 'test_data.csv'
    
    training_df.to_csv(train_path, index=False)
    test_df.to_csv(validation_path, index=False)
    print(f"Training data: {train_path.relative_to(root_dir)}")
    print(f"Columns: {training_df.shape[1]} (features + price)")
    print(f"Test data: {validation_path.relative_to(root_dir)}")
    print(f"Columns: {test_df.shape[1]} (features + price)")
    
    print("2. Copying to activationBase_german_car...")
    activation_path = root_dir / 'images' / 'activationBase_german_car' / 'activation_data.csv'
    activation_sample.to_csv(activation_path, index=False)
    print(f"Activation data: {activation_path.relative_to(root_dir)}")
    print(f"Columns: {activation_sample.shape[1]} features (NO price)")
    print(f"Has price column: {'price' in activation_sample.columns}")
    
    print("3. Copying to knowledgeBase_german_car...")
    output_dir = root_dir / 'output'
    knowledge_dir = root_dir / 'images' / 'knowledgeBase_german_car'
    
    files_to_copy = [
        ('currentAiSolution.pkl', 'currentAiSolution.pkl'),
        ('currentOlsSolution.pkl', 'currentOlsSolution.pkl'),
        ('scaler.pkl', 'scaler.pkl')
    ]
    
    for src_file, dst_file in files_to_copy:
        src_path = output_dir / src_file
        dst_path = knowledge_dir / dst_file
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"{src_file} → {dst_path.relative_to(root_dir)}")
        else:
            print(f"{src_file} not found in output directory")
    
    print("4. Copying visualization to codeBase_german_car...")
    png_src = root_dir / 'output' / 'model_diagnostics.png'
    png_dst = root_dir / 'images' / 'codeBase_german_car' / 'model_diagnostics.png'
    
    if png_src.exists():
        shutil.copy2(png_src, png_dst)
        print(f"model_diagnostics.png → {png_dst.relative_to(root_dir)}")
    else:
        print(f"model_diagnostics.png not found")

def create_scenario_docker_files(root_dir):
    print("Creating scenario docker-compose files...")
    
    ann_scenario_dir = root_dir / 'scenarios' / 'apply_annSolution_german_car' / 'x86_64'
    ann_scenario_dir.mkdir(parents=True, exist_ok=True)
    
    ann_compose_content = """services:
  knowledge:
    image: hassanimam7214/knowledgebase_german_car
    volumes:
      - ai_system_new:/tmp

  activation:
    image: hassanimam7214/activationbase_german_car
    volumes:
      - ai_system_new:/tmp

  code:
    image: hassanimam7214/codebase_german_car
    volumes:
      - ai_system_new:/tmp

volumes:
  ai_system_new:
    external: true
"""
    
    with open(ann_scenario_dir / 'docker-compose.yml', 'w') as f:
        f.write(ann_compose_content)
    
    print(f"ANN scenario docker-compose: {ann_scenario_dir.relative_to(root_dir)}")
    
    ols_scenario_dir = root_dir / 'scenarios' / 'apply_olsSolution_german_car' / 'x86_64'
    ols_scenario_dir.mkdir(parents=True, exist_ok=True)
    
    ols_compose_content = """services:
  knowledge:
    image: hassanimam7214/knowledgebase_german_car
    volumes:
      - ai_system_new:/tmp

  activation:
    image: hassanimam7214/activationbase_german_car
    volumes:
      - ai_system_new:/tmp

  code:
    image: hassanimam7214/codebase_german_car
    volumes:
      - ai_system_new:/tmp

volumes:
  ai_system_new:
    external: true
"""
    
    with open(ols_scenario_dir / 'docker-compose.yml', 'w') as f:
        f.write(ols_compose_content)
    
    print(f"OLS scenario docker-compose: {ols_scenario_dir.relative_to(root_dir)}")

def verify_data_formats(root_dir):
    print("Verifying data formats...")
    
    try:
        train_path = root_dir / 'output' / 'training_data.csv'
        train_df = pd.read_csv(train_path)
        train_has_price = 'price' in train_df.columns
        train_feature_cols = [col for col in train_df.columns if col != 'price']
        
        act_path = root_dir / 'output' / 'activation_data.csv'
        act_df = pd.read_csv(act_path)
        act_has_price = 'price' in act_df.columns
        act_feature_cols = list(act_df.columns)
        
        print(f"File verification:")
        print(f"Training data: {len(train_df):,} rows, {train_df.shape[1]} columns")
        print(f"Has price column: {train_has_price} (should be True)")
        print(f"Activation data: {len(act_df):,} rows, {act_df.shape[1]} columns")
        print(f"Has price column: {act_has_price} (should be False)")
        
        if set(train_feature_cols) == set(act_feature_cols):
            print(f"Feature columns match between training and activation data!")
            print(f"Features: {len(train_feature_cols)}")
        else:
            print(f"Feature columns DO NOT match!")
            train_only = set(train_feature_cols) - set(act_feature_cols)
            act_only = set(act_feature_cols) - set(train_feature_cols)
            if train_only:
                print(f"In training only: {train_only}")
            if act_only:
                print(f"In activation only: {act_only}")
            
        required_files = [
            root_dir / 'output' / 'joint_data_collection.csv',
            root_dir / 'output' / 'training_data.csv',
            root_dir / 'output' / 'test_data.csv',
            root_dir / 'output' / 'activation_data.csv',
            root_dir / 'output' / 'currentAiSolution.pkl',
            root_dir / 'output' / 'currentOlsSolution.pkl',
            root_dir / 'output' / 'scaler.pkl',
            root_dir / 'output' / 'model_diagnostics.png'
        ]
        
        print(f"Checking all output files exist:")
        all_exist = True
        for file_path in required_files:
            exists = file_path.exists()
            status = "✓" if exists else "✗"
            print(f"{status} {file_path.relative_to(root_dir)}")
            if not exists:
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

def main():
    print("=" * 70)
    print("GERMAN USED CAR PRICE PREDICTION - FINAL PROJECT")
    print("University of Potsdam - AI-CPS Compliant")
    print("=" * 70)
    
    print("[1] DETERMINING ROOT DIRECTORY")
    print("-" * 50)
    root_dir = get_root_directory()
    
    print("[2] CREATING AI-CPS DIRECTORY STRUCTURE")
    print("-" * 50)
    create_directories(root_dir)
    
    print("[3] LOADING DATA")
    print("-" * 50)
    df = load_data(root_dir)
    
    print("[4] CLEANING DATA")
    print("-" * 50)
    df_clean, initial_count, cleaned_count, cleaning_percentage = clean_data(df)
    df = df_clean
    
    print("[5] ENGINEERING FEATURES")
    print("-" * 50)
    X, y, y_original, encoded_features = engineer_features(df)
    
    print("[6] SPLITTING AND SCALING DATA")
    print("-" * 50)
    (X_train, X_test, y_train, y_test, y_train_orig, y_test_orig,
     X_train_scaled, X_test_scaled, scaler) = split_and_scale_data(X, y, y_original)
    
    scaler_path = root_dir / 'output' / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path.relative_to(root_dir)}")
    
    print("[7] SAVING CSV FILES")
    print("-" * 50)
    training_df, test_df, activation_sample = save_csv_files(
        X, y_original, X_train, y_train_orig, X_test, y_test_orig, 
        encoded_features, root_dir
    )
    
    print("[8] TRAINING OLS MODEL")
    print("-" * 50)
    ols_model, ols_mae, ols_rmse, ols_r2, y_pred_ols = train_ols_model(
        X_train_scaled, y_train, X_test_scaled, y_test_orig, root_dir
    )
    
    print("[9] TRAINING ANN MODEL")
    print("-" * 50)
    ann_model, ann_mae, ann_rmse, ann_r2, y_pred_ann = train_ann_model(
        X_train_scaled, y_train, X_test_scaled, y_test_orig, root_dir
    )
    
    print("[10] CREATING VISUALIZATIONS")
    print("-" * 50)
    create_visualizations(y_test_orig, y_pred_ols, y_pred_ann, 
                         ols_r2, ann_r2, ols_mae, ann_mae, root_dir)
    create_complete_comparison_plot(ols_mae, ols_rmse, ols_r2, ann_mae, ann_rmse, ann_r2, root_dir)

    
    print("[11] COPYING TO DOCKER DIRECTORIES")
    print("-" * 50)
    copy_files_to_images(training_df, test_df, activation_sample, root_dir)
    
    print("[12] CREATING SCENARIO DOCKER FILES")
    print("-" * 50)
    create_scenario_docker_files(root_dir)
    
    print("[13] FINAL VERIFICATION")
    print("-" * 50)
    verification_passed = verify_data_formats(root_dir)
    
    print("=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("PERFORMANCE SUMMARY:")
    print("-" * 50)
    print(f"{'Metric':<15} {'OLS':<15} {'ANN':<15} {'Winner':<10}")
    print(f"{'MAE (€)':<15} {ols_mae:<15.0f} {ann_mae:<15.0f} {'ANN' if ann_mae < ols_mae else 'OLS'}")
    print(f"{'RMSE (€)':<15} {ols_rmse:<15.0f} {ann_rmse:<15.0f} {'ANN' if ann_rmse < ols_rmse else 'OLS'}")
    print(f"{'R²':<15} {ols_r2:<15.4f} {ann_r2:<15.4f} {'ANN' if ann_r2 > ols_r2 else 'OLS'}")
    
    print("DATA SUMMARY:")
    print("-" * 50)
    print(f"Initial samples: {initial_count:,}")
    print(f"Cleaned samples: {cleaned_count:,} ({cleaning_percentage:.1f}% retained)")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features created: {len(encoded_features)}")
    
    print("FILES CREATED IN ROOT DIRECTORY:")
    print("-" * 50)
    print(f"/output/ - All CSV, models, and visualizations")
    print(f"/images/learningBase_german_car/ - Training data")
    print(f"/images/activationBase_german_car/ - Activation data")
    print(f"/images/knowledgeBase_german_car/ - Models")
    print(f"/images/codeBase_german_car/ - Visualizations")
    print(f"/scenarios/apply_annSolution_german_car/ - ANN deployment")
    print(f"/scenarios/apply_olsSolution_german_car/ - OLS deployment")
    
    if verification_passed:
        print(f"All verifications passed! Ready for AI-CPS deployment.")
    else:
        print(f"Some issues detected. Please review the verification output.")

if __name__ == "__main__":
    main()

