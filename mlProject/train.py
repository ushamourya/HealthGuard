
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/acutelines_synthetic.csv"

print("\n" + "=" * 70)
print("HOSPITAL EMERGENCY RISK PREDICTION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Number of records:", len(df))
print("Number of attributes:", len(df.columns))

# ============================================================
# 2. DISPLAY DATA INFORMATION
# ============================================================

print("\nDataset columns:")
print(df.columns.tolist())

print("\nRisk distribution:")
print(df["risk_level"].value_counts())

# ============================================================
# 3. REMOVE UNNECESSARY COLUMNS
# ============================================================

# patient_id is only an identifier
# triage_score is removed because it was used to create
# the target risk level and would cause data leakage.

X = df.drop(
    columns=[
        "risk_level",
        "patient_id",
        "triage_score"
    ]
)

y = df["risk_level"]

# ============================================================
# 4. ENCODE TARGET LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nRisk classes:")
for index, class_name in enumerate(label_encoder.classes_):
    print(index, "=", class_name)

# ============================================================
# 5. IDENTIFY COLUMN TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ============================================================
# 7. PREPROCESSING
# ============================================================

# Numerical preprocessing:
# 1. Fill missing values with median
# 2. Standardize numerical features

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# Categorical preprocessing:
# 1. Fill missing values
# 2. Convert categories into numerical values

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            __import__(
                "sklearn"
            ).preprocessing.OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)

# ============================================================
# 8. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )
}

# ============================================================
# 9. CREATE RESULT FOLDER
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

best_model = None
best_model_name = None
best_f1 = 0

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    # Create complete ML pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = pipeline.predict(
        X_test
    )

    # Evaluation
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # Save results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    # Save every trained model
    safe_name = model_name.lower().replace(
        " ",
        "_"
    )

    joblib.dump(
        pipeline,
        f"models/{safe_name}.pkl"
    )

    print(
        f"\nSaved model: models/{safe_name}.pkl"
    )

    # Find best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = model_name

# ============================================================
# 11. SAVE LABEL ENCODER
# ============================================================

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model:", best_model_name)
print("F1-Score:", round(best_f1, 4))

# ============================================================
# 13. SAVE COMPARISON RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel comparison:")
print(results_df)

print(
    "\nSaved results to:"
    " results/model_comparison.csv"
)

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)
