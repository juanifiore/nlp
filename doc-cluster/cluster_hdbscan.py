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


## RESUMEN DEL CODIGO
#model = SentenceTransformer("dunzhang/stella_en_400M_v5")
#docs = df['text'].tolist()
#embeddings = model.encode(docs)
#
#hdb = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=250, metric='euclidean')
#hdb.fit(embeddings)
#
#umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
#embeddings_dimension_2 = umap.fit_transform(embeddings)
#
#labels = hdb.labels_


# =====================
# STELLA EMBEDDING MODEL
# ====================
print('CARGANDO MODELO STELLA 400M')
#model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
model = SentenceTransformer(
     "dunzhang/stella_en_400M_v5",
     trust_remote_code=True,
     device="cpu",
     config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
 )
print('CALCULANDO EMBEDDINGS....')
embeddings = model.encode(docs)
# guardo los embeddings
#np.save('topic_sentences.npy', embeddings)
print('EMBEDDINGS CALCULADOS')

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
# indicar si usar metrica del coseno (implementada aparte) o una de hdbscan
metrica_coseno = False 

if metrica_coseno == False:
    print('CALCULANDO CLUSTERS SOBRE LOS EMBEDDINGS CON HDBSCAN ....')
    hdb = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=250, metric='euclidean').fit(embeddings)
    print('CLUSTERS CALCULADOS')
else:
    print('CALCULANDO CLUSTERS SOBRE LOS EMBEDDINGS CON HDBSCAN (COSENO)....')
    from sklearn.preprocessing import normalize
    embeddings_normalized = normalize(embeddings)
    hdb = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=10, metric='cosine').fit(embeddings_normalized)
    print('CLUSTERS CALCULADOS (COSENO)')


# ============================================
# REDUCCION DE LA DIMENSION PARA VISUALIZACION 
# ============================================

print('USANDO UMAP PARA REDUCCION DE DIMENSION....')
umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
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
