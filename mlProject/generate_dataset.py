
import os
import numpy as np
import pandas as pd

# ============================================================
# HOSPITAL EMERGENCY RISK PREDICTION
# SYNTHETIC DATASET GENERATOR
# ============================================================

np.random.seed(42)

NUMBER_OF_PATIENTS = 10000

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# ============================================================
# PATIENT INFORMATION
# ============================================================

patient_id = [
    f"P{i:05d}"
    for i in range(1, NUMBER_OF_PATIENTS + 1)
]

age = np.random.randint(
    18,
    91,
    NUMBER_OF_PATIENTS
)

gender = np.random.choice(
    ["Male", "Female"],
    NUMBER_OF_PATIENTS
)

# ============================================================
# VITAL SIGNS
# ============================================================

heart_rate = np.clip(
    np.random.normal(85, 20, NUMBER_OF_PATIENTS),
    45,
    180
).round().astype(int)

systolic_bp = np.clip(
    np.random.normal(125, 25, NUMBER_OF_PATIENTS),
    70,
    210
).round().astype(int)

diastolic_bp = np.clip(
    np.random.normal(78, 15, NUMBER_OF_PATIENTS),
    40,
    130
).round().astype(int)

respiratory_rate = np.clip(
    np.random.normal(18, 5, NUMBER_OF_PATIENTS),
    8,
    45
).round().astype(int)

oxygen_saturation = np.clip(
    np.random.normal(96, 3, NUMBER_OF_PATIENTS),
    75,
    100
).round(1)

temperature = np.clip(
    np.random.normal(37.0, 0.8, NUMBER_OF_PATIENTS),
    34,
    41.5
).round(1)

# ============================================================
# BIOCHEMICAL FEATURES
# ============================================================

glucose = np.clip(
    np.random.normal(115, 40, NUMBER_OF_PATIENTS),
    50,
    400
).round(1)

lactate = np.clip(
    np.random.normal(1.8, 0.8, NUMBER_OF_PATIENTS),
    0.5,
    8
).round(2)

# ============================================================
# SYMPTOMS
# ============================================================

chest_pain = np.random.choice(
    ["Yes", "No"],
    NUMBER_OF_PATIENTS,
    p=[0.30, 0.70]
)

shortness_of_breath = np.random.choice(
    ["Yes", "No"],
    NUMBER_OF_PATIENTS,
    p=[0.25, 0.75]
)

altered_mental_status = np.random.choice(
    ["Yes", "No"],
    NUMBER_OF_PATIENTS,
    p=[0.10, 0.90]
)

# ============================================================
# MEDICAL HISTORY
# ============================================================

comorbidity = np.random.choice(
    [
        "None",
        "Diabetes",
        "Hypertension",
        "Cardiac",
        "Respiratory",
        "Multiple"
    ],
    NUMBER_OF_PATIENTS,
    p=[
        0.35,
        0.15,
        0.18,
        0.10,
        0.10,
        0.12
    ]
)

# ============================================================
# ECG
# ============================================================

ecg_abnormality = np.random.choice(
    ["Normal", "Abnormal"],
    NUMBER_OF_PATIENTS,
    p=[0.70, 0.30]
)

# ============================================================
# CALCULATE SYNTHETIC TRIAGE SCORE
# ============================================================

score = np.zeros(
    NUMBER_OF_PATIENTS,
    dtype=int
)

# Heart rate
score += np.where(
    heart_rate > 120,
    2,
    np.where(heart_rate > 100, 1, 0)
)

# Low blood pressure
score += np.where(
    systolic_bp < 90,
    3,
    np.where(systolic_bp < 100, 2, 0)
)

# Very high blood pressure
score += np.where(
    systolic_bp > 180,
    2,
    0
)

# Respiratory rate
score += np.where(
    respiratory_rate > 30,
    3,
    np.where(respiratory_rate > 22, 1, 0)
)

# Oxygen saturation
score += np.where(
    oxygen_saturation < 90,
    4,
    np.where(oxygen_saturation < 94, 2, 0)
)

# Temperature
score += np.where(
    (temperature < 35) | (temperature > 39.5),
    2,
    np.where(
        (temperature < 36) | (temperature > 38.5),
        1,
        0
    )
)

# Glucose
score += np.where(
    (glucose < 60) | (glucose > 300),
    2,
    0
)

# Lactate
score += np.where(
    lactate > 4,
    3,
    np.where(lactate > 2.5, 1, 0)
)

# Symptoms
score += np.where(
    chest_pain == "Yes",
    1,
    0
)

score += np.where(
    shortness_of_breath == "Yes",
    1,
    0
)

score += np.where(
    altered_mental_status == "Yes",
    3,
    0
)

# Comorbidity
score += np.where(
    comorbidity == "Multiple",
    2,
    np.where(
        comorbidity != "None",
        1,
        0
    )
)

# ECG
score += np.where(
    ecg_abnormality == "Abnormal",
    2,
    0
)

# Age
score += np.where(
    age >= 75,
    2,
    np.where(age >= 60, 1, 0)
)

# ============================================================
# RISK CLASSIFICATION
# ============================================================

def calculate_risk(score):

    if score <= 3:
        return "Low"

    elif score <= 7:
        return "Medium"

    else:
        return "High"


risk_level = [
    calculate_risk(value)
    for value in score
]

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "patient_id": patient_id,

    "age": age,

    "gender": gender,

    "heart_rate": heart_rate,

    "systolic_bp": systolic_bp,

    "diastolic_bp": diastolic_bp,

    "respiratory_rate": respiratory_rate,

    "oxygen_saturation": oxygen_saturation,

    "temperature": temperature,

    "glucose": glucose,

    "lactate": lactate,

    "chest_pain": chest_pain,

    "shortness_of_breath": shortness_of_breath,

    "altered_mental_status": altered_mental_status,

    "comorbidity": comorbidity,

    "ecg_abnormality": ecg_abnormality,

    "triage_score": score,

    "risk_level": risk_level
})

# ============================================================
# ADD SOME MISSING VALUES
# ============================================================

for column in [
    "heart_rate",
    "systolic_bp",
    "oxygen_saturation",
    "glucose",
    "lactate"
]:

    missing_indices = np.random.choice(
        df.index,
        size=200,
        replace=False
    )

    df.loc[
        missing_indices,
        column
    ] = np.nan

# ============================================================
# SAVE DATASET
# ============================================================

output_file = (
    "dataset/acutelines_synthetic.csv"
)

df.to_csv(
    output_file,
    index=False
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "SYNTHETIC ACUTELINES DATASET CREATED"
)

print("=" * 60)

print(
    "\nFile:",
    output_file
)

print(
    "\nNumber of patients:",
    len(df)
)

print(
    "Number of attributes:",
    len(df.columns)
)

print("\nRisk distribution:")

print(
    df["risk_level"].value_counts()
)

print("\nDataset preview:")

print(
    df.head()
)

print("\nMissing values:")

print(
    df.isnull().sum()
)

print(
    "\nDataset creation completed successfully!"
)
