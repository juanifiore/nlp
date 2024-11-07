import os
import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP

#=======================================
# LIBRERIAS PARA DISTINTOS EMBEDDERS
#=======================================
# ----------------------------------------------
# descomentar para usar sentence-transformer  
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

df = pd.read_csv("frases-medicina-psico-matem.csv")
#df = df.sample(frac=0.05, random_state=42)

docs = df['description'].tolist()
#docs = [
#    f"{title}\n{description}"
#    for title, description in zip(df.title, df.description)
#]
#================================================


# =====================
# STELLA EMBEDDING MODEL
# ====================
print('Cargando modelo Stella 400M')
#model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
model = SentenceTransformer(
     "dunzhang/stella_en_400M_v5",
     trust_remote_code=True,
     device="cpu",
     config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
 )
print('Calculando embeddings....')
embeddings = model.encode(docs)
print('Embeddings calculados')

# =====================
# SENTECE TRANSFORMER MODEL
#======================= 
#model = SentenceTransformer('all-MiniLM-L6-v2')
#embeddings = model.encode(docs)

# =====================
# OPENAI MODEL
#======================= 
#client = OpenAI()
#response = client.embeddings.create(input=docs, model="text-embedding-3-small")
#embeddings = [np.array(x.embedding) for x in response.data]

# ======================
# CLUSTERING CON HDB
# =====================
print('CALCULANDO CLUSTERS SOBRE LOS EMBEDDINGS CON HDBSCAN ....')
hdb = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=3).fit(embeddings)
print('CLUSTERS CALCULADOS')

print('USANDO UMAP PARA REDUCCION DE DIMENSION....')
umap = UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)
df_umap = (
    pd.DataFrame(umap.fit_transform(np.array(embeddings)), columns=['x', 'y'])
    .assign(cluster=lambda df: hdb.labels_.astype(str))
    #.query('cluster != "-1"')  # descomentar para eliminar los documentos considerados 'ruido' del grafico
    .sort_values(by='cluster')
)

# ----------------------------------------------------------------------------
# columna para el tamaño del punto
df_umap['point_size'] = 3  # Ajusta este valor según el tamaño que prefieras
print('GRAFICANDO CLUSTERS...')
fig = px.scatter(df_umap, x='x', y='y', color='cluster', size='point_size', size_max=20)
fig.show()
# ----------------------------------------------------------------------------

#fig = px.scatter(df_umap, x='x', y='y', color='cluster')
#fig.show()
