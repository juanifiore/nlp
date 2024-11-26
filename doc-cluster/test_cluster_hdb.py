import os
import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from sklearn.preprocessing import normalize


# CARGAR EMBEDDINGS .npy 

embeddings = np.load('embeddings/7500_ag_news.npy')

# NORMALIZAR EMBEDDINGS 
#embeddings = normalize(embeddings, norm='l2')

# =============================
# REDUCCION DE DIMENSION PREVIA A CLUSTERIZAR 
# =============================

umap = UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1, metric = 'cosine')
embeddings = umap.fit_transform(embeddings)



# ======================
# CLUSTERING CON HDB
# =====================
# indicar si usar metrica del coseno (implementada aparte) o una de hdbscan
metrica_coseno = False 

if metrica_coseno == False:
    print('CALCULANDO CLUSTERS SOBRE LOS EMBEDDINGS CON HDBSCAN ....')
    hdb = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=200, metric='euclidean').fit(embeddings)
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
df_umap = (
    pd.DataFrame((embeddings), columns=['x', 'y'])
    .assign(cluster=lambda df: hdb.labels_.astype(str))
    #.query('cluster != "-1"')  # descomentar para eliminar los documentos considerados 'ruido' del grafico
    .sort_values(by='cluster')
)
# ----------------------------------------------------------------------------
# columna para el tamaño del punto
print('GRAFICANDO CLUSTERS...')
fig = px.scatter(df_umap, x='x', y='y', color='cluster')
fig.show()
# ----------------------------------------------------------------------------

# ============================================
# REDUCCION DE LA DIMENSION PARA VISUALIZACION 
# ============================================
#
#print('USANDO UMAP PARA REDUCCION DE DIMENSION....')
#umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric = 'cosine')
#df_umap = (
#    pd.DataFrame(umap.fit_transform(np.array(embeddings)), columns=['x', 'y'])
#    .assign(cluster=lambda df: hdb.labels_.astype(str))
#    #.query('cluster != "-1"')  # descomentar para eliminar los documentos considerados 'ruido' del grafico
#    .sort_values(by='cluster')
#)
## ----------------------------------------------------------------------------
## columna para el tamaño del punto
#print('GRAFICANDO CLUSTERS...')
#fig = px.scatter(df_umap, x='x', y='y', color='cluster')
#fig.show()
## ----------------------------------------------------------------------------
