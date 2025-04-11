# Visualizations made using ChatGPT
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load processed data for features, generated in Part1.py
df = pd.read_csv('car_cleaned.csv')

# Drop class
df = df.drop(columns=['class'])


# Model 1: K-Means
kValues = [2, 3, 4, 5, 6]
silScoreKmeans = []

for k in kValues:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df)
    score = silhouette_score(df, labels)
    silScoreKmeans.append(score)
    print(f'k = {k}. Silhouette score: {score:.4f}')

plt.figure(figsize=(8, 5))
sns.lineplot(x=kValues, y=silScoreKmeans, marker='o')
plt.title('Silhouette Score for Different K (K-Means)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/kmeans_silhouette_scores.png')
plt.close()


# Model 2: Hierarchical
linkageMethods = ['ward', 'average', 'complete']
silScoreHier = []

for method in linkageMethods:
    linked = linkage(df, method=method)
    plt.figure(figsize=(8, 4))
    dendrogram(linked, truncate_mode='lastp', p=20)
    plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f'plots/dendrogram_{method}.png')
    plt.close()

    clusterLabels = fcluster(linked, t=4, criterion='maxclust')
    score = silhouette_score(df, clusterLabels)
    silScoreHier.append(score)
    print(f'{method.capitalize()} linkage silhouette score: {score:.4f}')

plt.figure(figsize=(8, 5))
sns.lineplot(x=linkageMethods, y=silScoreHier, marker='o')
plt.title('Silhouette Score for Different Linkage Methods')
plt.xlabel('Linkage Method')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/hierarchical_silhouette_scores.png')
plt.close()