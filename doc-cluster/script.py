import os

from sentence_transformers import SentenceTransformer

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP

# descomentar si se usa el embedder de openai (se necesita key de api en un archivo .env dentro del directorio)
#import requests
#from dotenv import load_dotenv
#from openai import OpenAI

load_dotenv()

df = pd.read_csv("news_data_dedup.csv")
df = df.sample(frac=0.05, random_state=42)

docs = [
    f"{title}\n{description}"
    for title, description in zip(df.title, df.description)
]

# =====================
# SENTECE TRANSFORMER MODEL
#======================= 
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(docs)


# =====================
# OPENAI MODEL
#======================= 
#client = OpenAI()
#response = client.embeddings.create(input=docs, model="text-embedding-3-small")
#embeddings = [np.array(x.embedding) for x in response.data]


# ======================
# CLUSTERING CON HDB
# =====================
hdb = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3).fit(embeddings)

umap = UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)

df_umap = (
    pd.DataFrame(umap.fit_transform(np.array(embeddings)), columns=['x', 'y'])
    .assign(cluster=lambda df: hdb.labels_.astype(str))
    .query('cluster != "-1"')
    .sort_values(by='cluster')
)

# Añadir una columna para el tamaño del punto
df_umap['point_size'] = 10  # Ajusta este valor según el tamaño que prefieras

fig = px.scatter(df_umap, x='x', y='y', color='cluster', size='point_size', size_max=20)
fig.show()

#fig = px.scatter(df_umap, x='x', y='y', color='cluster')
#fig.show()
