import pandas as pd
import math
import json

# Load the dataset and fuzzy membership functions
df = pd.read_csv("C:/Users/Andrew/Hackaton/weather_with_fuzzy_label_prediction(new).csv")
with open("C:/Users/Andrew/Hackaton/fuzzy_inference_system/fuzzy_membership_functions.json") as f:
    membership_data = json.load(f)

# Extract parameters for PRCP (precipitation)
prcp_info = membership_data["features"]["prcp"]
min_val = prcp_info["normalization"]["min"]
max_val = prcp_info["normalization"]["max"]
mf_list = prcp_info["membership_functions"]

# Define Gaussian membership function
def gaussmf(x, c, sigma):
    return math.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# Normalize function
def normalize(value):
    return (value - min_val) / (max_val - min_val)

# Apply labeling based on membership functions
def get_prcp_label(prcp_val):
    norm_val = normalize(prcp_val)
    memberships = [gaussmf(norm_val, mf["center"], mf["sigma"]) for mf in mf_list]
    labels = ["Low", "Medium", "High"]
    return labels[memberships.index(max(memberships))]

# Apply the function to the dataset
df["prcp_label_from_function"] = df["prcp"].apply(get_prcp_label)

# Save the result
df.to_csv("C:/Users/Andrew/Hackaton/labeled_prcp_output.csv", index=False)
