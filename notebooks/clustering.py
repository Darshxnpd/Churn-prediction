import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("C:/Code/Churn-Prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix potential issues with 'TotalCharges' column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])  # Remove rows with missing values

# Feature selection
features = df[['MonthlyCharges', 'TotalCharges', 'tenure']]

# Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled)

# Dimensionality Reduction
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)

# Plot
plt.scatter(reduced[:, 0], reduced[:, 1], c=df['Segment'], cmap='viridis')
plt.title("Customer Segments (KMeans Clustering)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Segment")
plt.show()
