
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# ============================================================
# HOSPITAL EMERGENCY RISK PREDICTION
# VISUALIZATION DASHBOARD
# ============================================================

print("\n" + "=" * 70)
print("HOSPITAL EMERGENCY RISK PREDICTION")
print("MODEL EVALUATION DASHBOARD")
print("=" * 70)

# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)

# ============================================================
# LOAD MODEL RESULTS
# ============================================================

results = pd.read_csv(
    "results/model_comparison.csv"
)

print("\nMODEL PERFORMANCE")
print("-" * 70)

print(results.to_string(index=False))

# ============================================================
# FIND BEST MODEL
# ============================================================

best_row = results.loc[
    results["F1_Score"].idxmax()
]

best_model_name = best_row["Model"]
best_accuracy = best_row["Accuracy"] * 100
best_precision = best_row["Precision"] * 100
best_recall = best_row["Recall"] * 100
best_f1 = best_row["F1_Score"] * 100

print("\nBEST MODEL")
print("-" * 70)

print("Model     :", best_model_name)
print("Accuracy  :", f"{best_accuracy:.2f}%")
print("Precision :", f"{best_precision:.2f}%")
print("Recall    :", f"{best_recall:.2f}%")
print("F1-Score  :", f"{best_f1:.2f}%")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "dataset/acutelines_synthetic.csv"
)

# ============================================================
# RISK DISTRIBUTION
# ============================================================

risk_counts = (
    df["risk_level"]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"]
    )
)

# ============================================================
# DASHBOARD
# ============================================================

sns.set_theme(
    style="whitegrid"
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10)
)

fig.suptitle(
    "Hospital Emergency Risk Prediction\n"
    "Machine Learning Model Evaluation",
    fontsize=20,
    fontweight="bold"
)

# ============================================================
# GRAPH 1 - ACCURACY
# ============================================================

ax1 = axes[0, 0]

bars = ax1.bar(
    results["Model"],
    results["Accuracy"] * 100,
    color=[
        "#e74c3c",
        "#2ecc71",
        "#3498db"
    ]
)

ax1.set_title(
    "Accuracy Comparison",
    fontsize=14,
    fontweight="bold"
)

ax1.set_ylabel(
    "Accuracy (%)"
)

ax1.set_ylim(
    0,
    100
)

for bar in bars:

    value = bar.get_height()

    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

# ============================================================
# GRAPH 2 - PRECISION / RECALL / F1
# ============================================================

ax2 = axes[0, 1]

x = range(
    len(results)
)

width = 0.25

ax2.bar(
    [i - width for i in x],
    results["Precision"] * 100,
    width,
    label="Precision",
    color="#9b59b6"
)

ax2.bar(
    x,
    results["Recall"] * 100,
    width,
    label="Recall",
    color="#f39c12"
)

ax2.bar(
    [i + width for i in x],
    results["F1_Score"] * 100,
    width,
    label="F1-Score",
    color="#1abc9c"
)

ax2.set_title(
    "Precision, Recall and F1-Score",
    fontsize=14,
    fontweight="bold"
)

ax2.set_ylabel(
    "Score (%)"
)

ax2.set_xticks(
    list(x)
)

ax2.set_xticklabels(
    results["Model"]
)

ax2.set_ylim(
    0,
    100
)

ax2.legend()

# ============================================================
# GRAPH 3 - RISK DISTRIBUTION
# ============================================================

ax3 = axes[1, 0]

bars = ax3.bar(
    risk_counts.index,
    risk_counts.values,
    color=[
        "#2ecc71",
        "#f1c40f",
        "#e74c3c"
    ]
)

ax3.set_title(
    "Emergency Risk Class Distribution",
    fontsize=14,
    fontweight="bold"
)

ax3.set_xlabel(
    "Risk Level"
)

ax3.set_ylabel(
    "Number of Patients"
)

for bar in bars:

    value = bar.get_height()

    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        value + 50,
        str(int(value)),
        ha="center",
        fontweight="bold"
    )

# ============================================================
# GRAPH 4 - BEST MODEL
# ============================================================

ax4 = axes[1, 1]

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

metric_values = [
    best_accuracy,
    best_precision,
    best_recall,
    best_f1
]

bars = ax4.barh(
    metric_names,
    metric_values,
    color=[
        "#3498db",
        "#9b59b6",
        "#f39c12",
        "#1abc9c"
    ]
)

ax4.set_title(
    f"Best Model: {best_model_name}",
    fontsize=14,
    fontweight="bold"
)

ax4.set_xlabel(
    "Performance (%)"
)

ax4.set_xlim(
    0,
    100
)

for bar in bars:

    value = bar.get_width()

    ax4.text(
        value + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}%",
        va="center",
        fontweight="bold"
    )

# ============================================================
# DASHBOARD LAYOUT
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94
    ]
)

# ============================================================
# SAVE DASHBOARD
# ============================================================

dashboard_file = (
    "results/ml_dashboard.png"
)

plt.savefig(
    dashboard_file,
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nDashboard saved to:",
    dashboard_file
)

# ============================================================
# SHOW DASHBOARD
# ============================================================

print(
    "\nOpening dashboard..."
)

plt.show()

# ============================================================
# CONFUSION MATRICES
# ============================================================

print(
    "\nGenerating confusion matrices..."
)

X = df.drop(
    columns=[
        "risk_level",
        "patient_id",
        "triage_score"
    ]
)

y = df["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load label encoder

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

# Load models

models = {

    "Decision Tree":
        joblib.load(
            "models/decision_tree.pkl"
        ),

    "Random Forest":
        joblib.load(
            "models/random_forest.pkl"
        ),

    "XGBoost":
        joblib.load(
            "models/xgboost.pkl"
        )
}

numeric_labels = [
    0,
    1,
    2
]

class_names = [
    "High",
    "Low",
    "Medium"
]

# ============================================================
# CONFUSION MATRIX WINDOWS
# ============================================================

for model_name, model in models.items():

    predictions = model.predict(
        X_test
    )

    predictions = predictions.astype(int)

    actual_numeric = label_encoder.transform(
        y_test
    )

    matrix = confusion_matrix(
        actual_numeric,
        predictions,
        labels=numeric_labels
    )

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        f"{model_name}\nConfusion Matrix",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Predicted Risk Level"
    )

    plt.ylabel(
        "Actual Risk Level"
    )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        f"results/{filename}",
        dpi=300
    )

    print(
        f"Created: {filename}"
    )

    # Display the confusion matrix
    plt.show()

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print(
    "VISUALIZATION COMPLETED SUCCESSFULLY"
)

print("=" * 70)

print(
    "\nYour graphs are available in:"
)

print(
    "results/"
)

print(
    "\nClose the graph windows when finished."
)

print("=" * 70)

