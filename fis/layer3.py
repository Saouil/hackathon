import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import numpy as np

# Load weather data and normalization info
df = pd.read_csv("C:/Users/Andrew/Hackaton/weather_data_with_year_month.csv")
with open("C:/Users/Andrew/Hackaton/fuzzy_inference_system/fuzzy_membership_functions.json", "r") as f:
    fuzzy_mfs = json.load(f)
with open("C:/Users/Andrew/Hackaton/fuzzy_inference_system/fuzzy_rules_layer2_clean.json", "r") as f:
    symbolic_rules = json.load(f)["rules"]

features = list(fuzzy_mfs["features"].keys())
output_feature = "prcp"

# Normalize input features
for feature in features:
    min_val = fuzzy_mfs["features"][feature]["normalization"]["min"]
    max_val = fuzzy_mfs["features"][feature]["normalization"]["max"]
    df[f"{feature}_norm"] = (df[feature] - min_val) / (max_val - min_val)

# Generate TSK rules with linear regression per rule
tsk_rules = []
X_all = df[[f"{f}_norm" for f in features]].values
y_all = df[output_feature].values

for rule in symbolic_rules:
    condition = rule["IF"]

    # Find rows that roughly match the rule's fuzzy labels (simple threshold logic)
    mask = np.ones(len(df), dtype=bool)
    for feat in features:
        label = condition[feat]
        label_index = ['Low', 'Medium', 'High'].index(label)
        center = fuzzy_mfs["features"][feat]["membership_functions"][label_index]["center"]
        sigma = fuzzy_mfs["features"][feat]["membership_functions"][label_index]["sigma"]

        # Approximate fuzzy matching: keep samples where feature_norm is near center
        lower = center - sigma
        upper = center + sigma
        mask &= (df[f"{feat}_norm"] >= lower) & (df[f"{feat}_norm"] <= upper)

    # Subset data for this rule
    X_sub = X_all[mask]
    y_sub = y_all[mask]

    # Skip if too few samples
    if len(X_sub) < 1:
        continue

    # Train linear regression for this rule
    reg = LinearRegression()
    reg.fit(X_sub, y_sub)
    coefficients = reg.coef_.tolist()
    intercept = reg.intercept_

    tsk_rules.append({
        "IF": condition,
        "THEN": {
            "equation": f"{intercept:.4f} + " + " + ".join([f"{coef:.4f}*{feat}" for coef, feat in zip(coefficients, features)]),
            "intercept": intercept,
            "coefficients": dict(zip(features, coefficients))
        }
    })

# Save the TSK rules
tsk_json_path = "C:/Users/Andrew/Hackaton/fuzzy_inference_system/tsk_rules_layer3.json"
with open(tsk_json_path, "w") as f:
    json.dump({"rules": tsk_rules}, f, indent=4)

tsk_json_path
