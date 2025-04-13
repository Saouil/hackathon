import json
import itertools

# Load membership function data (Layer 1)
with open("C:/Users/Andrew/Hackaton/fuzzy_inference_system/fuzzy_membership_functions.json", "r") as f:
    layer1_data = json.load(f)

features = list(layer1_data["features"].keys())
term_labels = ['Low', 'Medium', 'High']

# Reconstruct fuzzy terms without assigning output (Layer 2 clean rule base)
fuzzy_terms = {}
for feature in features:
    centers = [mf['center'] for mf in layer1_data["features"][feature]["membership_functions"]]
    fuzzy_terms[feature] = list(zip(term_labels, centers))

# Generate symbolic fuzzy rules only (Layer 2)
fuzzy_rules = []
for combination in itertools.product(term_labels, repeat=len(features)):
    rule = {
        "IF": {features[i]: combination[i] for i in range(len(features))}
    }
    fuzzy_rules.append(rule)

# Save updated fuzzy rules (Layer 2) to JSON
layer2_clean_path = "C:/Users/Andrew/Hackaton/fuzzy_inference_system/fuzzy_rules_layer2_clean.json"
with open(layer2_clean_path, "w") as f:
    json.dump({"rules": fuzzy_rules}, f, indent=4)

layer2_clean_path
