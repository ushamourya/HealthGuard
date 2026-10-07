# HealthGuard 🏥

 HealthGuard is a machine learning-based emergency risk prediction system that classifies patients into Low, Medium, or High Risk based on clinical parameters.

  The project uses a synthetic dataset and compares Decision Tree, Random Forest, and XGBoost models to select the best-performing model.

## 🎯 Objective

The objective of HealthGuard is to demonstrate how machine learning
can be used to classify patient emergency risk based on clinical
parameters and provide a simple risk prediction system.

# 🛠️ Technologies

- Python

- Pandas

- NumPy

- Scikit-learn

- XGBoost

- Matplotlib

- Seaborn

- Joblib



# ⚙️ Installation

```
Clone the repository:
git clone https://github.com/your-username/HealthGuard.git
```

```
cd HealthGuard
```

# Install dependencies:

```
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn

```

### Run the files in this order:

### 1. Generate Dataset

```
python generate_dataset.py

```
### 2. Train Models

```
python train.py

```
### 3. Generate Results

```
python visualize_result.py

```
### 4. Run the Application

```
python app.py

```

### You can then choose:

 1. Existing Patient
 2. New Patient
 3. Exit


### The system predicts:

- Low Risk
- Medium Risk
- High Risk

# 📊 Machine Learning Models

## The project compares:

### Model	Purpose
- Decision Tree	Tree-based classification
- Random Forest	Ensemble classification
- XGBoost	Gradient boosting classification

### The best model is selected using weighted F1-score.

# 👨‍💻 Conclusion
HealthGuard demonstrates how machine learning can be applied to healthcare-related classification problems. By comparing Decision Tree, Random Forest, and XGBoost models, the project shows how different algorithms can be evaluated and how the best-performing model can be integrated into a practical prediction system.

The project provides a foundation for exploring machine learning, healthcare analytics, model evaluation, and predictive systems in a simple and understandable way.
