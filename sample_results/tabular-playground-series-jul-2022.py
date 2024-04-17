import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("./input/data.csv")
data.drop("id", axis=1, inplace=True)

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Determine the optimal number of clusters using the elbow method and silhouette score
range_n_clusters = list(range(2, 11))
silhouette_scores = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Select the number of clusters with the highest silhouette score
optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]

# Fit the KMeans model with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
data["Predicted"] = kmeans.fit_predict(data_scaled)

# Evaluate the model using silhouette score
final_silhouette_score = silhouette_score(data_scaled, data["Predicted"])
print(
    f"Silhouette Score for optimal clusters {optimal_n_clusters}: {final_silhouette_score}"
)

# Save the predictions to a CSV file
submission = pd.read_csv("./input/sample_submission.csv")
submission["Predicted"] = data["Predicted"]
submission.to_csv("./working/submission.csv", index=False)
