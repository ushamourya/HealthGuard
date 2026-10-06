
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================

# EXISTING + NEW PATIENT
# ============================================================

print("\n" + "=" * 65)
print("HealthGuard System")
print("=" * 65)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load("models/best_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    print("\nModel loaded successfully!")
    print("Best Model: XGBoost")

except Exception as e:
    print("\nERROR: Could not load model.")
    print(e)
    raise SystemExit


# ============================================================
# LOAD DATASET
# ============================================================

dataset_path = "dataset/data_synthetic.csv"

if os.path.exists(dataset_path):
    dataset = pd.read_csv(dataset_path)
else:
    dataset = None


# ============================================================
# INPUT HELPER FUNCTIONS
# ============================================================

def get_number(prompt):
    """
    Accepts numerical input.

    User can:
    - Enter a number
    - Leave blank
    - Enter NA / N/A / Unknown / None

    Missing values become NaN.
    """

    while True:
        value = input(prompt).strip()

        if value == "" or value.lower() in [
            "na",
            "n/a",
            "unknown",
            "none"
        ]:
            return float("nan")

        try:
            return float(value)

        except ValueError:
            print("Invalid number. Please enter a valid number.")
            print("You can also leave it blank if unavailable.")


def get_text(prompt):
    """
    CASE-INSENSITIVE TEXT INPUT.

    Examples:

    male / Male / MALE / mAlE -> Male
    female / Female / FEMALE -> Female

    yes / Yes / YES -> Yes
    no / No / NO -> No

    normal / Normal / NORMAL -> Normal
    abnormal / Abnormal / ABNORMAL -> Abnormal

    cardiac / Cardiac / CARDIAC -> Cardiac

    Blank / NA / N/A / Unknown -> Unknown
    """

    while True:

        value = input(prompt).strip()

        # Missing value
        if value == "" or value.lower() in [
            "na",
            "n/a",
            "unknown"
        ]:
            return "Unknown"

        value_lower = value.lower()

        # ----------------------------------------------------
        # Gender
        # ----------------------------------------------------

        if value_lower == "male":
            return "Male"

        if value_lower == "female":
            return "Female"

        # ----------------------------------------------------
        # Yes / No
        # ----------------------------------------------------

        if value_lower == "yes":
            return "Yes"

        if value_lower == "no":
            return "No"

        # ----------------------------------------------------
        # ECG
        # ----------------------------------------------------

        if value_lower == "normal":
            return "Normal"

        if value_lower == "abnormal":
            return "Abnormal"

        # ----------------------------------------------------
        # Comorbidity
        # ----------------------------------------------------

        if value_lower == "none":
            return "None"

        if value_lower == "diabetes":
            return "Diabetes"

        if value_lower == "hypertension":
            return "Hypertension"

        if value_lower == "cardiac":
            return "Cardiac"

        if value_lower == "respiratory":
            return "Respiratory"

        if value_lower == "multiple":
            return "Multiple"

        # ----------------------------------------------------
        # Invalid input
        # ----------------------------------------------------

        print("\nInvalid input.")

        print("Please enter one of the accepted values.")
        print("")


# ============================================================
# MENU
# ============================================================

print("\nSelect Patient Input Method")
print("-" * 65)
print("1. Existing Patient from Dataset")
print("2. New Patient")
print("3. Exit")

choice = input("\nEnter choice (1/2/3): ").strip().lower()


# Allow words as well as numbers

if choice in [
    "1",
    "existing",
    "existing patient",
    "old"
]:
    choice = "1"

elif choice in [
    "2",
    "new",
    "new patient"
]:
    choice = "2"

elif choice in [
    "3",
    "exit",
    "quit"
]:
    choice = "3"


# ============================================================
# EXIT
# ============================================================

if choice == "3":

    print("\nProgram closed.")
    raise SystemExit


# ============================================================
# EXISTING PATIENT
# ============================================================

if choice == "1":

    if dataset is None:

        print("\nDataset not found!")

        print(
            "Please make sure "
            "dataset/acutelines_synthetic.csv exists."
        )

        raise SystemExit

    print("\nExisting Patient Search")
    print("-" * 65)

    patient_id = input(
        "Enter Patient ID (example: P00001): "
    ).strip().upper()

    patient_rows = dataset[
        dataset["patient_id"].astype(str).str.upper()
        == patient_id
    ]

    if patient_rows.empty:

        print("\nPatient ID not found.")

        print(
            "Please select option 2 to enter a new patient."
        )

        raise SystemExit

    patient = patient_rows.iloc[0]

    print("\nPatient found successfully!")
    print("-" * 65)

    print("Patient ID:", patient["patient_id"])

    # --------------------------------------------------------
    # FEATURES USED DURING TRAINING
    # --------------------------------------------------------

    feature_columns = [

        "age",
        "gender",
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "respiratory_rate",
        "oxygen_saturation",
        "temperature",
        "glucose",
        "lactate",
        "chest_pain",
        "shortness_of_breath",
        "altered_mental_status",
        "comorbidity",
        "ecg_abnormality"

    ]

    patient_data = pd.DataFrame(
        [patient[feature_columns].to_dict()]
    )


# ============================================================
# NEW PATIENT
# ============================================================

elif choice == "2":

    print("\nNew Patient Details")
    print("-" * 65)

    age = get_number(
        "Age: "
    )

    gender = get_text(
        "Gender (Male/Female): "
    )

    heart_rate = get_number(
        "Heart Rate (bpm) [blank if unavailable]: "
    )

    systolic_bp = get_number(
        "Systolic Blood Pressure (mmHg) [blank if unavailable]: "
    )

    diastolic_bp = get_number(
        "Diastolic Blood Pressure (mmHg) [blank if unavailable]: "
    )

    respiratory_rate = get_number(
        "Respiratory Rate (breaths/min) [blank if unavailable]: "
    )

    oxygen_saturation = get_number(
        "Oxygen Saturation (%) [blank if unavailable]: "
    )

    temperature = get_number(
        "Temperature (°C) [blank if unavailable]: "
    )

    glucose = get_number(
        "Glucose (mg/dL) [blank if unavailable]: "
    )

    lactate = get_number(
        "Lactate (mmol/L) [blank if unavailable]: "
    )

    chest_pain = get_text(
        "Chest Pain (Yes/No) [blank if unavailable]: "
    )

    shortness_of_breath = get_text(
        "Shortness of Breath (Yes/No) [blank if unavailable]: "
    )

    altered_mental_status = get_text(
        "Altered Mental Status (Yes/No) [blank if unavailable]: "
    )

    comorbidity = get_text(
        "Comorbidity "
        "(None/Diabetes/Hypertension/Cardiac/Respiratory/Multiple): "
    )

    ecg_abnormality = get_text(
        "ECG Abnormality (Normal/Abnormal) "
        "[blank if unavailable]: "
    )

    # --------------------------------------------------------
    # CREATE PATIENT DATAFRAME
    # --------------------------------------------------------

    patient_data = pd.DataFrame({

        "age": [age],

        "gender": [gender],

        "heart_rate": [heart_rate],

        "systolic_bp": [systolic_bp],

        "diastolic_bp": [diastolic_bp],

        "respiratory_rate": [respiratory_rate],

        "oxygen_saturation": [oxygen_saturation],

        "temperature": [temperature],

        "glucose": [glucose],

        "lactate": [lactate],

        "chest_pain": [chest_pain],

        "shortness_of_breath": [
            shortness_of_breath
        ],

        "altered_mental_status": [
            altered_mental_status
        ],

        "comorbidity": [comorbidity],

        "ecg_abnormality": [
            ecg_abnormality
        ]

    })


else:

    print("\nInvalid choice.")

    print(
        "Please enter 1, 2, or 3."
    )

    raise SystemExit


# ============================================================
# DISPLAY PATIENT DATA
# ============================================================

print("\n" + "=" * 65)
print("PATIENT INFORMATION")
print("=" * 65)

print(
    patient_data.to_string(index=False)
)


# ============================================================
# PREDICTION
# ============================================================

try:

    prediction = model.predict(
        patient_data
    )

except Exception as e:

    print("\nPrediction error:")
    print(e)

    print(
        "\nThe model could not process the entered data."
    )

    print(
        "Please check that the input values match "
        "the training data format."
    )

    raise SystemExit


# ============================================================
# CONVERT PREDICTION TO RISK NAME
# ============================================================

try:

    predicted_class = label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]

except Exception:

    predicted_class = str(prediction[0])


# ============================================================
# PREDICTION PROBABILITY
# ============================================================

try:

    probabilities = model.predict_proba(
        patient_data
    )[0]

    class_numbers = model.classes_

    probability_dict = {}

    for class_number, probability in zip(
        class_numbers,
        probabilities
    ):

        class_name = label_encoder.inverse_transform(
            [int(class_number)]
        )[0]

        probability_dict[class_name] = (
            probability * 100
        )

except Exception as e:

    print("\nProbability calculation unavailable.")

    probability_dict = {
        "High": 0,
        "Medium": 0,
        "Low": 0
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 65)
print("EMERGENCY RISK PREDICTION")
print("=" * 65)

print(
    "\nPredicted Risk Level:",
    predicted_class.upper()
)


# ============================================================
# RISK MESSAGE
# ============================================================

if predicted_class == "High":

    print("\n⚠ HIGH RISK")

    print(
        "Patient may require immediate medical attention."
    )

elif predicted_class == "Medium":

    print("\n⚠ MEDIUM RISK")

    print(
        "Patient requires clinical monitoring "
        "and timely evaluation."
    )

else:

    print("\n✓ LOW RISK")

    print(
        "Patient is predicted to have a lower emergency risk."
    )


# ============================================================
# DISPLAY PROBABILITIES
# ============================================================

print("\nRisk Prediction Probability")
print("-" * 65)

for risk in [
    "High",
    "Medium",
    "Low"
]:

    probability = probability_dict.get(
        risk,
        0
    )

    print(
        f"{risk:<10}: {probability:.2f}%"
    )


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# GRAPH 1
# RISK PROBABILITY
# ============================================================

risk_names = [
    "Low",
    "Medium",
    "High"
]

risk_probabilities = [

    probability_dict.get(
        "Low",
        0
    ),

    probability_dict.get(
        "Medium",
        0
    ),

    probability_dict.get(
        "High",
        0
    )

]

colors = [
    "#2ecc71",
    "#f1c40f",
    "#e74c3c"
]


plt.figure(
    figsize=(9, 6)
)

bars = plt.bar(
    risk_names,
    risk_probabilities,
    color=colors
)

plt.title(
    "Patient Emergency Risk Probability",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Risk Level"
)

plt.ylabel(
    "Probability (%)"
)

plt.ylim(
    0,
    100
)


for bar in bars:

    value = bar.get_height()

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        value + 2,

        f"{value:.2f}%",

        ha="center",

        fontweight="bold"

    )


plt.tight_layout()

plt.savefig(
    "results/patient_risk_probability.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 2
# PATIENT CLINICAL MEASUREMENTS
# ============================================================

measurement_names = [

    "Heart Rate",
    "Systolic BP",
    "Respiratory Rate",
    "Oxygen Saturation",
    "Temperature",
    "Glucose",
    "Lactate"

]


measurement_values = [

    patient_data["heart_rate"].iloc[0],

    patient_data["systolic_bp"].iloc[0],

    patient_data["respiratory_rate"].iloc[0],

    patient_data["oxygen_saturation"].iloc[0],

    patient_data["temperature"].iloc[0],

    patient_data["glucose"].iloc[0],

    patient_data["lactate"].iloc[0]

]


# ============================================================
# REFERENCE RANGES
# Used only for visualization
# ============================================================

normal_min = [

    60,
    90,
    12,
    95,
    36.1,
    70,
    0.5

]

normal_max = [

    100,
    120,
    20,
    100,
    37.2,
    140,
    2.0

]


normalized_values = []
display_values = []


for value, minimum, maximum in zip(

    measurement_values,
    normal_min,
    normal_max

):

    if pd.isna(value):

        normalized_values.append(0)

        display_values.append("N/A")

    else:

        normalized = (

            (value - minimum)
            / (maximum - minimum)

        ) * 100

        normalized = max(
            0,
            min(
                normalized,
                150
            )
        )

        normalized_values.append(
            normalized
        )

        display_values.append(
            value
        )


# ============================================================
# CREATE CLINICAL GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

bars = plt.bar(

    measurement_names,

    normalized_values,

    color="#3498db"

)


plt.axhline(

    0,

    color="green",

    linestyle="--",

    label="Normal Range Lower Boundary"

)


plt.axhline(

    100,

    color="red",

    linestyle="--",

    label="Normal Range Upper Boundary"

)


plt.title(

    "Patient Clinical Parameter Analysis",

    fontsize=16,

    fontweight="bold"

)


plt.xlabel(
    "Clinical Parameter"
)

plt.ylabel(
    "Relative Level (%)"
)


plt.xticks(
    rotation=25
)


for bar, value in zip(

    bars,
    display_values

):

    height = bar.get_height()

    if value == "N/A":

        text = "N/A"

    else:

        text = f"{value:g}"

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        height + 3,

        text,

        ha="center",

        fontsize=9,

        fontweight="bold"

    )


plt.legend()

plt.tight_layout()


plt.savefig(

    "results/patient_clinical_analysis.png",

    dpi=300

)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)

print("PATIENT ANALYSIS COMPLETED")

print("=" * 65)


print(
    f"\nPredicted Risk: {predicted_class.upper()}"
)


print("\nRisk Probabilities:")

for risk in [
    "High",
    "Medium",
    "Low"
]:

    print(
        f"{risk:<10}: "
        f"{probability_dict.get(risk, 0):.2f}%"
    )


print("\nGraphs saved:")

print(
    "1. results/patient_risk_probability.png"
)

print(
    "2. results/patient_clinical_analysis.png"
)

print(
    "\nClose the graph windows when finished."
)

print("=" * 65)
