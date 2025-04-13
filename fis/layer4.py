import json
import pandas as pd
import numpy as np

# Load data and membership functions
with open("C:/Users/Andrew/Hackaton/fuzzy_inference_system/fuzzy_membership_functions.json", "r") as f:
    fuzzy_mfs = json.load(f)

with open("C:/Users/Andrew/Hackaton/fuzzy_inference_system/tsk_rules_layer3.json", "r") as f:
    tsk_rules = json.load(f)["rules"]

df = pd.read_csv("C:/Users/Andrew/Hackaton/weather_data_with_year_month.csv")

features = list(fuzzy_mfs["features"].keys())
output_feature = "prcp"

# Normalize inputs using Layer 1 normalization info
for feature in features:
    min_val = fuzzy_mfs["features"][feature]["normalization"]["min"]
    max_val = fuzzy_mfs["features"][feature]["normalization"]["max"]
    df[f"{feature}_norm"] = (df[feature] - min_val) / (max_val - min_val)

# Gaussian membership function
def gaussian_mf(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# New fuzzy labels for prcp (used as categorical output instead of raw value)
fuzzy_output_labels = ['Low', 'Medium', 'High']
prcp_mfs = fuzzy_mfs["features"]["prcp"]["membership_functions"]
prcp_label_mapping = list(zip(fuzzy_output_labels, prcp_mfs))

# Perform fuzzy inference using fuzzy output labels instead of crisp prcp prediction
def infer_label(row, rules, fuzzy_info):
    weighted_outputs = {label: 0.0 for label, _ in prcp_label_mapping}
    total_weights = {label: 0.0 for label, _ in prcp_label_mapping}

    for rule in rules:
        firing_strength = 1.0
        for feat, level in rule["IF"].items():
            label_index = ['Low', 'Medium', 'High'].index(level)
            center = fuzzy_info["features"][feat]["membership_functions"][label_index]["center"]
            sigma = fuzzy_info["features"][feat]["membership_functions"][label_index]["sigma"]
            x = row[f"{feat}_norm"]
            mu = gaussian_mf(x, center, sigma)
            firing_strength *= mu

        if firing_strength == 0:
            continue

        # Compute crisp TSK output
        output = rule["THEN"]["intercept"]
        for feat, coef in rule["THEN"]["coefficients"].items():
            output += coef * row[f"{feat}_norm"]

        # Fuzzify the TSK output using prcp membership functions
        for label, mf in prcp_label_mapping:
            mu_out = gaussian_mf((output - fuzzy_info["features"]["prcp"]["normalization"]["min"]) /
                                 (fuzzy_info["features"]["prcp"]["normalization"]["max"] -
                                  fuzzy_info["features"]["prcp"]["normalization"]["min"]),
                                 mf["center"], mf["sigma"])
            weighted_outputs[label] += mu_out * firing_strength
            total_weights[label] += firing_strength

    # Choose label with highest normalized score
    fuzzy_scores = {label: weighted_outputs[label] / total_weights[label]
                    if total_weights[label] > 0 else 0.0
                    for label in weighted_outputs}
    return max(fuzzy_scores, key=fuzzy_scores.get)

# Apply label-based inference across dataset
df['fuzzy_prcp_label'] = df.apply(lambda row: infer_label(row, tsk_rules, fuzzy_mfs), axis=1)

# Save labeled prediction result
inferred_label_path = "C:/Users/Andrew/Hackaton/weather_with_fuzzy_label_prediction(new).csv"
df.to_csv(inferred_label_path, index=False)

inferred_label_path
