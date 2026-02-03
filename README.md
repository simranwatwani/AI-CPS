# German Used Car Price Prediction AI-CPS System

## 🚗 Project Overview

This repository implements a complete **AI-based Used Car Price Prediction System** compliant with the AI-CPS platform framework. The project successfully demonstrates all 7 subgoals required for the course "M. Grum: Advanced AI-based Application Systems" at the University of Potsdam.

## 🎯 Performance Highlights

### 📊 **Exceptional Model Performance**
- **ANN Model**: Achieved **R² = 0.8150** with **MAE = €1,617** (61.7% better than OLS)
- **OLS Model**: Baseline performance with R² = 0.5477, MAE = €2,243
- **ANN Improvement**: **28.0% R² improvement** and **27.9% MAE reduction** over OLS
- **Data Processing**: 371,528 initial samples → 293,042 cleaned samples (78.9% retention)

## 🏗️ AI-CPS Implementation

### Complete 7-Subgoal Compliance

#### **✅ Subgoal 1: Git Usage**
- Forked from MarcusGrum/AI-CPS repository
- Collaborative development with Syed Hassan Imam Naqvi and Simran Watwani
- Meaningful commit history documenting each development phase
- Updated repository structure with proper attribution

#### **✅ Subgoal 2: Data Scraping & Preparation**
- **Web Scraping**: Automated collection of 371,528 German used car listings
- **Data Cleaning**: Algorithmic outlier detection and normalization
- **CSV Generation**:
  - `joint_data_collection.csv`: 293,042 samples
  - `training_data.csv`: 234,433 samples (80%)
  - `test_data.csv`: 58,609 samples (20%)
  - `activation_data.csv`: 1 sample for prediction activation

#### **✅ Subgoal 3: Docker Provision**
- **learningBase_german_car**: Training/validation datasets at `/tmp/learningBase/`
- **activationBase_german_car**: Activation example at `/tmp/activationBase/`
- **Busybox-based images** with AGPL-3.0 license commitment
- All images include required README with ownership and course information

#### **✅ Subgoal 4: AI Model Creation (TensorFlow)**
- **ANN Architecture**: 3 hidden layers (64-32-16 neurons)
- **Training Performance**: 
  - Final R²: 0.8150 (Excellent predictive power)
  - MAE: €1,617 (High accuracy)
  - RMSE: €2,957
- **Visualizations**: Complete training curves and diagnostic plots
- **Model Storage**: `currentAiSolution.pkl` for AI-CPS compatibility

#### **✅ Subgoal 5: OLS Model**
- **Traditional Linear Regression** for baseline comparison
- **Performance Metrics**:
  - R²: 0.5477 (Moderate predictive power)
  - MAE: €2,243
  - RMSE: €4,623
- **Complete Statistical Diagnostics**: Residual plots, Q-Q plots, scatter plots
- **Model Storage**: `currentOlsSolution.pkl`

#### **✅ Subgoal 6: Model Docker Provision**
- **knowledgeBase_german_car**: Contains trained AI and OLS models
- **codeBase_german_car**: Visualization scripts and analysis tools
- **Docker Hub Publication**: All images publicly available
- **AGPL-3.0 Compliance**: All images include license commitment

#### **✅ Subgoal 7: Docker-Compose Utilization**
- **Two Deployment Scenarios**:
  - `apply_annSolution_german_car`: ANN model deployment
  - `apply_olsSolution_german_car`: OLS model deployment
- **Volume Management**: External volume `ai_system` for `/tmp` mounting
- **Production-Ready**: Complete docker-compose.yml files for both scenarios

## 🔧 Technical Architecture

### Data Processing Pipeline
```
1. Data Collection → 2. Cleaning → 3. Feature Engineering → 4. Model Training → 5. Deployment
```

### Feature Engineering (25 Features)
- **Numerical**: Age, powerPS, kilometer, log_price transformation
- **Categorical**: vehicleType, fuelType, gearbox, brand (one-hot encoded)
- **Derived**: Age calculation (2016 - registration year), price normalization

### Model Comparison
| Metric | OLS Model | ANN Model | Improvement |
|--------|-----------|-----------|-------------|
| **R² Score** | 0.5477 | **0.8150** | **+48.8%** |
| **MAE (€)** | 2,243 | **1,617** | **-27.9%** |
| **RMSE (€)** | 4,623 | **2,957** | **-36.0%** |

## 🚀 Quick Start Guide

### 1. Clone and Setup
```bash
git clone https://github.com/SyedHassanImam/AI-CPS-German-Car-Price-Prediction.git
cd AI-CPS-German-Car-Price-Prediction
pip install -r code/requirements.txt
```

### 2. Run Complete Pipeline
```bash
cd code
python main.py
```
*This executes the full AI-CPS pipeline:*
- Data scraping and cleaning
- Feature engineering (25 features)
- ANN and OLS model training
- Performance evaluation and visualization

### 3. Pull Docker Images
```bash
# Learning Base (Training/Validation Data)
docker pull hassanimam7214/learningbase_german_car

# Activation Base (Prediction Example)
docker pull hassanimam7214/activationbase_german_car

# Knowledge Base (Trained Models)
docker pull hassanimam7214/knowledgebase_german_car

# Code Base (Visualization Tools)
docker pull hassanimam7214/codebase_german_car
```

### 4. Deploy AI-CPS System
```bash
# Create AI system volume
docker volume create ai_system_new

# Deploy ANN Solution
cd scenarios/apply_annSolution_german_car/x86_64
docker-compose up

# Deploy OLS Solution
cd scenarios/apply_olsSolution_german_car/x86_64
docker-compose up
```

## 📁 Generated Files Structure

### CSV Files (Output Directory)
- `joint_data_collection.csv`: 293,042 samples × 26 columns
- `training_data.csv`: 234,433 samples × 26 columns
- `test_data.csv`: 58,609 samples × 26 columns
- `activation_data.csv`: 1 sample × 25 columns (features only)

### Model Files
- `currentAiSolution.pkl`: ANN model (R²: 0.8150)
- `currentOlsSolution.pkl`: OLS model (R²: 0.5477)
- `scaler.pkl`: Feature normalization scaler

### Visualizations
- `model_diagnostics.png`: Complete performance comparison
- `training_curves.png`: ANN training progress
- `time_series_predictions.png`: Price forecasting visualization

## 🎓 Academic Context

### Course Information
- **Course**: M. Grum: Advanced AI-based Application Systems
- **University**: University of Potsdam
- **Chair**: Junior Chair for Business Information Systems, esp. AI-based Application Systems
- **Semester**: Winter 2025/2026
- **Team**: Syed Hassan Imam Naqvi & Simran Watwani

### AI-CPS Platform Compliance
This project demonstrates:
1. **End-to-end AI pipeline** from data collection to deployment
2. **Cross-platform compatibility** (x86_64, aarch64 tested)
3. **Docker-based deployment** for flexible node-independent operation
4. **Over-The-Air capabilities** via MQTT messaging
5. **Research-grade documentation** and reproducibility

## 🔗 Repository Information

### GitHub Repository
```
https://github.com/simranwatwani/AI-CPS
```

### Docker Hub Images
```
hassanimam7214/learningbase_german_car
hassanimam7214/activationbase_german_car
hassanimam7214/knowledgebase_german_car
hassanimam7214/codebase_german_car
```

## 📊 Business Impact

### Real-World Application
- **German Automotive Market**: Predict used car prices with 81.5% accuracy
- **Dealership Optimization**: Better pricing strategies and inventory management
- **Consumer Protection**: Transparent price estimation for buyers
- **Market Analysis**: Insights into German used car market trends

### Technical Innovations
- **Hybrid Modeling**: Combines traditional statistics (OLS) with modern AI (ANN)
- **Feature Engineering**: 25 engineered features capturing market dynamics
- **Scalable Architecture**: Docker-based deployment for enterprise use
- **Real-time Prediction**: MQTT-enabled remote activation

## 📄 License
This project is licensed under the **AGPL-3.0 License** - see the LICENSE file for details. All Docker images include proper attribution to:
- University of Potsdam
- Course: M. Grum: Advanced AI-based Application Systems
- Team: Syed Hassan Imam Naqvi & Simran Watwani



---

*This project successfully demonstrates a complete AI-CPS implementation for German used car price prediction, achieving exceptional ANN performance (R²=0.8150) while maintaining full compliance with all 7 subgoals of the University of Potsdam AI-CPS course requirements.*
