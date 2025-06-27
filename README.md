# Insurance Charges Prediction Project (COSC 202)

## Project Overview
This project develops a neural network regression model to predict insurance charges based on customer attributes such as age, BMI, smoking status, and other factors. The goal is to build a comprehensive machine learning pipeline for data exploration, preprocessing, model training, evaluation, and deployment.

**Dataset:** 1,338 records with 7 columns (age, sex, bmi, children, smoker, region, charges)  
**Deadline:** June 29th, 2025  

## Table of Contents
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [The Data Science Process](#the-data-science-process)
- [Model Architecture](#model-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Saving and Loading Functionality](#saving-and-loading-functionality)
- [Performance Metrics](#performance-metrics)
- [Usage Instructions](#usage-instructions)
- [Dependencies](#dependencies)

## Project Structure
```
data_project/
│
├── Insurance_Charges_Prediction_Project.ipynb        # Jupyter notebook with all project functionality
├── insurance.csv           # Dataset
├── requirements.txt        # Project dependencies 
├── fix_dependencies.sh     # Script to fix dependency issues
├── project.txt             # Project requirements and description
├── model_artifacts/        # Generated directory for saved model artifacts
│   ├── insurance_model_TIMESTAMP.h5       # Trained Keras model
│   ├── preprocessor_TIMESTAMP.pkl         # Saved preprocessing pipeline
│   ├── training_history_TIMESTAMP.json    # Training metrics history
│   ├── metrics_TIMESTAMP.json             # Evaluation metrics
│   ├── hyperparameters_TIMESTAMP.json     # Best hyperparameters
│   └── manifest_TIMESTAMP.json            # Manifest tracking all artifacts
├── training_evaluation.png  # Enhanced training visualization
├── model_analysis.png       # Advanced model performance analysis
└── README.md               # Project documentation (this file)
```

## Technologies Used

### Core Libraries
- **TensorFlow/Keras**: Selected for building neural network models with high-level abstraction and comprehensive ecosystem
- **scikit-learn**: Used for data preprocessing, model evaluation, and hyperparameter tuning via GridSearchCV
- **pandas**: Employed for data manipulation and analysis
- **NumPy**: Utilized for numerical operations and array handling
- **Matplotlib/Seaborn**: Chosen for data visualization and model performance plotting

### Specialized Components
- **scikeras**: Adopted for scikit-learn compatibility with newer TensorFlow versions
- **tqdm**: Implemented for progress tracking throughout the pipeline
- **pickle**: Used for serializing the preprocessing pipeline
- **Jupyter Notebook**: For interactive development and documentation

### Justification of Technology Choices
1. **Neural Networks (over traditional ML)**: 
   - Better capture of complex non-linear relationships between customer attributes and insurance charges
   - More flexibility in model architecture design
   - Superior performance when properly tuned

2. **TensorFlow/Keras**:
   - Industry-standard framework for deep learning
   - High-level API makes model creation and training intuitive
   - Extensive functionality for callbacks, model checkpointing, and early stopping

3. **scikeras**:
   - Bridges scikit-learn and Keras for seamless integration
   - Ensures compatibility with GridSearchCV for hyperparameter tuning
   - Maintains sklearn's familiar API while using Keras models

4. **tqdm Progress Bars**:
   - Provides visual feedback for time-consuming operations
   - Improves user experience during lengthy training sessions
   - Enables better tracking of project execution

## The Data Science Process

### 1. Data Exploration and Visualization
The project begins with thorough exploration of the insurance dataset to understand:
- Data distributions and statistical properties
- Correlations between features
- Identification of key predictors (e.g., smoking status has the highest impact on charges)
- Detection of outliers and potential data quality issues

Various visualizations were created using Seaborn and Matplotlib to understand feature relationships:
- Correlation heatmap
- Distribution plots
- Scatter plots
- Box plots for categorical comparisons

### 2. Data Preprocessing
A robust preprocessing pipeline was implemented using scikit-learn's ColumnTransformer:

- **Numerical Features** (age, bmi, children):
  - Scaled using StandardScaler to normalize features to similar ranges

- **Categorical Features** (sex, smoker, region):
  - Encoded using OneHotEncoder with dropping of first category to avoid multicollinearity
  - Compatibility handling for different scikit-learn versions (sparse vs. sparse_output parameter)

- **Train-Test Split**:
  - Data divided into 80% training and 20% testing sets
  - Stratified splitting to maintain distribution of key variables

### 3. Model Development
A flexible neural network architecture was developed with:
- Configurable number of hidden layers
- Tunable neurons per layer
- Dropout layers for regularization
- ReLU activation for hidden layers
- Linear activation for output (regression task)
- Adam optimizer with configurable learning rate

### 4. Hyperparameter Tuning
Comprehensive hyperparameter tuning using GridSearchCV with 5-fold cross-validation:

**Parameters Tuned:**
- Hidden layers: [1, 2, 3]
- Neurons per layer: [32, 64, 128]
- Dropout rate: [0.1, 0.2, 0.3]
- Learning rate: [0.01, 0.001, 0.0001]
- Batch size: [16, 32, 64]
- Epochs: [50, 100]

The tuning process dynamically adapts to both legacy TensorFlow and newer scikeras implementations.

### 5. Model Training and Evaluation
The final model is trained with:
- Best hyperparameters from tuning
- Early stopping to prevent overfitting
- Model checkpointing to save best weights
- Custom progress tracking with tqdm

**Enhanced Visualizations:**
- Training and validation loss curves
- MAE progression over epochs
- Actual vs. predicted values scatter plot
- Prediction error distribution
- Residuals analysis plots
- Feature-specific error analysis

**Evaluation Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### 6. Model Serialization and Versioning
A comprehensive saving system was implemented that captures:
- Trained Keras model (HDF5 format)
- Preprocessing pipeline (pickle format)
- Training history (JSON)
- Evaluation metrics (JSON)
- Hyperparameters (JSON)
- Manifest file for versioning and tracking

## Model Architecture
The neural network architecture consists of:
```
Input Layer (features) → Dense Layer with ReLU → Dropout →
[Additional Hidden Layers with ReLU and Dropout] → Output Layer (1 neuron, linear)
```

**Key Components:**
- **Input Dimension**: Adapts to preprocessed feature count
- **Hidden Layers**: Variable (1-3) based on tuning
- **Neurons**: Variable (32-128) based on tuning
- **Dropout**: Applied for regularization (0.1-0.3)
- **Optimizer**: Adam with tunable learning rate

## Hyperparameter Tuning
The hyperparameter tuning approach addresses compatibility challenges between different versions of KerasRegressor:

1. **Dual Implementation**:
   - Automatically detects available implementations (scikit-learn vs. scikeras)
   - Adjusts parameter format accordingly

2. **Parameter Standardization**:
   - For scikeras: Uses `model__` prefix (e.g., `model__hidden_layers`)
   - For legacy: Uses direct parameter names (e.g., `hidden_layers`)

3. **Progress Monitoring**:
   - Custom progress bar for GridSearchCV execution
   - Estimates completion time for lengthy tuning process

## Saving and Loading Functionality
The project implements a robust model persistence strategy:

1. **Timestamped Artifacts**:
   - All saved files include timestamps for versioning
   - Enables tracking multiple model versions

2. **Complete Ecosystem Capture**:
   - Model weights and architecture
   - Preprocessing pipeline
   - Training history
   - Evaluation metrics
   - Hyperparameters

3. **Manifest System**:
   - Central JSON manifest tracks locations of all artifacts
   - Simplifies model loading and reconstruction

## Performance Metrics
The model's performance is evaluated using:

- **MSE**: Measures average squared difference between predictions and actual values
- **RMSE**: Interpreted in the same units as the target variable (dollars)
- **MAE**: Average absolute difference between predictions and actual values
- **R²**: Proportion of variance in charges explained by the model

## Usage Instructions

### Installation
```bash
# Clone repository
git clone <repository-url>
cd data_project

# Fix dependencies (if needed)
chmod +x fix_dependencies.sh
./fix_dependencies.sh

# Or install requirements directly
pip install -r requirements.txt
```

### Running the Project
```bash
# Open the Jupyter notebook
jupyter notebook Maeen\'s_AI.ipynb

# Or run with Jupyter Lab
jupyter lab
```

The notebook is divided into clear sections that follow the data science workflow:
1. Library imports and setup
2. Data exploration 
3. Data visualization
4. Preprocessing
5. Neural network model design
6. Hyperparameter tuning
7. Model training and evaluation
8. Model saving and persistence

### Making New Predictions
```python
# Example code for making predictions with saved model
# Add this code to a new cell in the notebook

# Load model from a specific manifest
manifest_path = 'model_artifacts/manifest_TIMESTAMP.json'
model, preprocessor = load_model_from_manifest(manifest_path)

# Create new data sample
new_data = pd.DataFrame({
    'age': [30],
    'sex': ['male'],
    'bmi': [25.5],
    'children': [2],
    'smoker': ['no'],
    'region': ['northeast']
})

# Make prediction
predicted_charge = make_prediction(new_data, model, preprocessor)
print(f"Predicted insurance charge: ${predicted_charge[0]:.2f}")
```

## Dependencies
The project requires specific package versions to ensure compatibility:

- numpy>=1.20.0,<1.24.0
- pandas>=1.3.0,<2.0.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- tensorflow>=2.8.0,<2.12.0
- scikeras>=0.9.0
- tqdm>=4.61.0

Version constraints handle compatibility issues between packages, particularly for TensorFlow and NumPy.

---

*This project was completed as part of the COSC 202 course requirements.*
