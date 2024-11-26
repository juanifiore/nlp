import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from umap import UMAP
import plotly.express as px

# RESUMEN DEL CODIGO
#df = pd.read_csv("datasets/topic_sentences.csv")
#docs = df['text'].tolist()
#
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#embeddings = model.encode(docs)
#
#kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#kmeans.fit(embeddings)
#
#umap = UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1, metric='cosine')
#embeddings_2d = umap.fit_transform(embeddings)
#
#df_umap = pd.DataFrame(embeddings_2d, columns=["x", "y"])
#df_umap["cluster"] = labels


#=======================================
# LIBRERIAS PARA DISTINTOS EMBEDDERS
#=======================================
# ----------------------------------------------
# descomentar para usar sentence-transformer (stella 400 M) 
from sentence_transformers import SentenceTransformer
# ----------------------------------------------
# descomentar si se usa el embedder de openai (se necesita key de api en un archivo .env dentro del directorio)
#import requests
#from openai import OpenAI
#from dotenv import load_dotenv
#load_dotenv()
# ----------------------------------------------

# ===============================================
# CARGAR DATASET Y CREAR LISTA 'docs' 
# ===============================================

print('CARGANDO DATASET....')
df = pd.read_csv("datasets/topic_sentences.csv")
#df = df.sample(frac=0.05, random_state=42)


print('CONVIRTIENDO DATASET A LISTA DE STRINGS....')
# descomentar lineas para el dataset a usar

#----------------------------------------------------------
# computacion
#df = df.iloc[:200]
#docs = df['sentence'].tolist()

#----------------------------------------------------------

# ----------------------------------------------------------
#----- IRIS
# sacar ultima columna donde indica tipo de flor
#df = df.iloc[:, :-1]

# convertir las caracteristicas numericas a texto sin palabras adicionales
#docs = df.astype(str).apply(" ".join, axis=1).tolist()

# convertir las caracteristicas numericas a texto con palabras adicionales
#docs = [
#    f"The flower has a sepal length of {row['sepal.length']} cm, "
#    f"sepal width of {row['sepal.width']} cm, "
#    f"petal length of {row['petal.length']} cm, "
#    f"and petal width of {row['petal.width']} cm."
#    for _, row in df.iterrows()
#]

# ----------------------------------------------------------
# ----- FRASES PSICO-MATE... --------
docs = df['text'].tolist()

# ----------------------------------------------------------
#-------- NEWS .... ------
#docs = [
#    f"{title}\n{description}"
#    for title, description in zip(df.title, df.description)
#]
# ----------------------------------------------------------
#================================================


# =====================
# STELLA EMBEDDING MODEL
# ====================
print('CARGANDO MODELO STELLA 400M')
#model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
#model = SentenceTransformer(
#     "dunzhang/stella_en_400M_v5",
#     trust_remote_code=True,
#     device="cpu",
#     config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
# )
#print('CALCULANDO EMBEDDINGS....')
#embeddings = model.encode(docs)
## guardo los embeddings
##np.save('topic_sentences.npy', embeddings)
#print('EMBEDDINGS CALCULADOS')

# =====================
# SENTECE TRANSFORMER MODEL
#======================= 
#model = SentenceTransformer('all-MiniLM-L6-v2')
#embeddings = model.encode(docs)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(docs)

# =====================
# OPENAI MODEL
#======================= 
#client = OpenAI()
#response = client.embeddings.create(input=docs, model="text-embedding-3-small")
#embeddings = [np.array(x.embedding) for x in response.data]


# =========================
# KMeans Clustering
# =========================
num_clusters = 7  # Number of topics/categories
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

