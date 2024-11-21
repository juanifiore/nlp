import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP
import pandas as pd
import plotly.express as px

# =========================
# Embeddings and Labels
# =========================
embeddings = np.load('embeddings/7500_ag_news.npy')

#cargo labels del df original para ver como grafica umap reduciendo la dimension
#df = pd.read_csv('datasets/ag_news_test.csv')
#labels = df['label'].tolist()

# =========================
# KMeans Clustering
# =========================
num_clusters = 4  # Number of topics/categories
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)

# Predicted cluster labels
labels = kmeans.labels_

# ESTABLECER PARAMETRO PARA DIMENSION 2 O 3
dimension_reducida = 2

if dimension_reducida == 2 :
    # =========================
    # UMAP Dimensionality Reduction
    # =========================
    umap = UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1, metric='cosine')
    embeddings_2d = umap.fit_transform(embeddings)

    # =========================
    # Create DataFrame for Visualization
    # =========================
    df_umap = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    #df_umap["cluster"] = cluster_labels.astype(str)
    df_umap["cluster"] = labels

    # =========================
    # Plot with Plotly
    # =========================
    fig = px.scatter(df_umap, x="x", y="y", color="cluster",
                     title="KMeans Clusters Visualized with UMAP",
                     color_discrete_sequence=px.colors.qualitative.Set1)

    fig.update_traces(marker=dict(size=5))
    fig.show()

else:

    # =========================
    # UMAP for 3D Dimensionality Reduction
    # =========================
    umap = UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.1, metric='cosine')
    embeddings_3d = umap.fit_transform(embeddings)

    # =========================
    # Create DataFrame for Visualization
    # =========================
    df_umap = pd.DataFrame(embeddings_3d, columns=["x", "y", "z"])
    df_umap["cluster"] = labels  # Assuming `labels` contains the cluster labels

    # =========================
    # Plot with Plotly in 3D
    # =========================
    fig = px.scatter_3d(df_umap, x="x", y="y", z="z", color="cluster",
                        title="KMeans Clusters Visualized with UMAP in 3D",
                        color_discrete_sequence=px.colors.qualitative.Set1)

    fig.update_traces(marker=dict(size=5))
    fig.show()
