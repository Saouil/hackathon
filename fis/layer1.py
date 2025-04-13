import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def gaussian_mf(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# Load dataset
df = pd.read_csv("C:/Users/Andrew/Hackaton/weather_data_with_year_month.csv")

# Select numerical features for clustering
features = ['tavg', 'tmin', 'tmax', 'prcp', 'pres']
data = df[features].copy()

# Normalize the data for clustering and store original min/max for reference
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
normalized_df = pd.DataFrame(data_scaled, columns=features)
scaling_info = {
    feature: {
        "min": float(scaler.data_min_[i]),
        "max": float(scaler.data_max_[i])
    }
    for i, feature in enumerate(features)
}

# Apply KMeans clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(data_scaled)
cluster_centers = kmeans.cluster_centers_

# Add cluster labels to the data
normalized_df['cluster'] = cluster_labels

# Directory to save plots
plot_dir = "C:/Users/Andrew/Hackaton/fuzzy_inference_system/membership_plots/"
os.makedirs(plot_dir, exist_ok=True)

# Dictionary to save membership functions and normalization info
membership_dict = {"features": {}}

x_vals = np.linspace(0, 1, 100)

for i, feature in enumerate(features):
    centers = cluster_centers[:, i]
    centers.sort()

    sigma = (max(centers) - min(centers)) / 4 if len(set(centers)) > 1 else 0.1

    membership_dict["features"][feature] = {
        "normalization": scaling_info[feature],
        "membership_functions": []
    }

    plt.figure()
    for j, c in enumerate(centers):
        y = gaussian_mf(x_vals, c, sigma)
        plt.plot(x_vals, y, label=f'Gaussian {j+1} (c={c:.2f}, σ={sigma:.2f})')
        membership_dict["features"][feature]["membership_functions"].append({
            "center": float(c),
            "sigma": float(sigma),
            "type": "gaussmf"
        })

    plt.title(f'Membership Functions for {feature}')
    plt.xlabel('Normalized Value')
    plt.ylabel('Membership Degree')
    plt.legend()
    plot_path = os.path.join(plot_dir, f"{feature}_membership.png")
    plt.savefig(plot_path)
    plt.close()

# Save membership function parameters to JSON
json_path = "C:/Users/Andrew/Hackaton/fuzzy_inference_system/fuzzy_membership_functions.json"
with open(json_path, "w") as f:
    json.dump(membership_dict, f, indent=4)

json_path, plot_dir
