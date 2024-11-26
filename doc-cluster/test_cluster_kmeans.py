import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP
import pandas as pd
import plotly.express as px


# =========================
# Embeddings and Labels
# =========================
embeddings = np.load('embeddings/embeddings_6topic.npy')


#cargo labels del df original para ver como grafica umap reduciendo la dimension
#df = pd.read_csv('datasets/ag_news_train.csv')
#df = df.iloc[:15000]

#-----------------------------------------------------------
# agrego una columna con labels con indices
# Crear un diccionario para mapear categorías a números
#label_to_number = {label: idx for idx, label in enumerate(df['label_str'].unique())}
#
## Agregar la nueva columna numérica al DataFrame
#df['label'] = df['label_str'].map(label_to_number)

#-----------------------------------------------------------

#labels_reales = df['label'].tolist()
labels_reales = np.load('embeddings/labels_6topic.npy')


# swaps de labels para allmini6 para que coincidan con los de kmeans
#for i in range(len(labels_reales)):
#    if labels_reales[i] == 1:
#        labels_reales[i] = 2
#    elif labels_reales[i] == 2:
#        labels_reales[i] = 1
#    if labels_reales[i] == 1:
#        labels_reales[i] = 3
#    elif labels_reales[i] == 3:
#        labels_reales[i] = 1

# swaps de labels para stella
#for i in range(len(labels_reales)):
#    if labels_reales[i] == 1:
#        labels_reales[i] = 2
#    elif labels_reales[i] == 2:
#        labels_reales[i] = 1



# =========================
# KMeans Clustering
# =========================
#num_clusters = 7  # Number of topics/categories
#kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#kmeans.fit(embeddings)
##
### Predicted cluster labels

# -----------------------------------------------------------

#from sklearn.preprocessing import StandardScaler, normalize
#from sklearn.decomposition import PCA

# Normalizar embeddings
#embeddings = StandardScaler().fit_transform(embeddings)

# (Opcional) Reducir dimensionalidad con PCA
#pca = PCA(n_components=50, random_state=42)
#embeddings = pca.fit_transform(embeddings)

# Configurar y ajustar KMeans
kmeans = KMeans(
    n_clusters=6,        # Número de clusters
    init='k-means++',    # Inicialización mejorada
    n_init=20,           # Más reinicios para evitar mínimos locales
    max_iter=500,        # Más iteraciones para convergencia
    tol=1e-4,            # Tolerancia ajustada
    random_state=42,     # Reproducibilidad
    #algorithm='auto'     # Algoritmo automático
)
kmeans.fit(embeddings)

labels = kmeans.labels_


# ==================================================
# REASIGNAMOS LABELS DE KMEANS PARA QUE SEAN SIMILARES A LOS REALES (PARA QUE TENGAN MISMOS COLORES EN EL GRAFICO)
# ==================================================
#from sklearn.metrics import confusion_matrix
#
## Obtener los labels reales
#labels_reales = df['label'].values
#
## Calcular la matriz de confusión
#conf_matrix = confusion_matrix(labels_reales, labels)
#
## Asignar cada cluster al label real más común en ese cluster
#cluster_to_label = {}
#for cluster in range(conf_matrix.shape[1]):
#    cluster_to_label[cluster] = np.argmax(conf_matrix[:, cluster])
#
## Reasignar los kmeans.labels_ usando la correspondencia
#labels = np.array([cluster_to_label[label] for label in kmeans.labels_])

# ----------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(labels_reales, labels)

# Usar Hungarian Algorithm para maximizar coincidencias
row_ind, col_ind = linear_sum_assignment(-conf_matrix)

# Crear un diccionario para mapear los clusters de KMeans a los labels reales
label_mapping = {kmeans_label: real_label for kmeans_label, real_label in zip(col_ind, row_ind)}

# Reasignar los labels de KMeans
labels_reasignados = np.array([label_mapping[label] for label in labels])



# ===================================
# METRICA PARA ACCURACY
# ===================================
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
conf_matrix = confusion_matrix(labels_reales, labels_reasignados)
accuracy = accuracy_score(labels_reales, labels_reasignados)

print("EMBEDDER: stella_en_400M_v5")
accuracy = accuracy_score(labels_reales, labels_reasignados)
print(f"Accuracy: {accuracy}")

# Mostrar reporte de clasificación
print("\nClassification Report:\n")
print(classification_report(labels_reales, labels_reasignados))

## Mostrar matriz de confusión
print("\nMatriz de confusion:\n")
print(conf_matrix)




# ESTABLECER PARAMETRO PARA DIMENSION 2 O 3
dimension_reducida = 2

if dimension_reducida == 2 :
    # =========================
    # UMAP Dimensionality Reduction
    # =========================
    n_neighbors = 200 
    min_dist = 0.05
    umap = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine')
    embeddings_2d = umap.fit_transform(embeddings)

    # =========================
    # Create DataFrame for Visualization
    # =========================
    df_umap = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    #df_umap["cluster"] = cluster_labels.astype(str)
    df_umap["cluster"] = labels_reales.astype(str)
    #df_umap["cluster"] = df['label_str'].tolist()

    # df para clusters de kmeans
    df_umap_kmeans = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df_umap_kmeans["cluster"] = labels_reasignados.astype(str)

    # =========================
    # Plot with Plotly
    # =========================
    #grafico clusters con labels reales
    color_map = {
    '0': 'red',
    '1': 'blue',
    '2': 'green',
    '3': 'purple',
    '4': 'orange',
    '5': 'brown',
}
    fig = px.scatter(df_umap, x="x", y="y", color="cluster", 
                     title="Clusters REALES , n_neighbors= " + str(n_neighbors) + ", min_dist= " + str(min_dist), 
                     color_discrete_map=color_map)

    fig.update_traces(marker=dict(size=5))
    fig.show()

    # grafico clusteres estimados por kmeans
    fig_kmeans = px.scatter(df_umap_kmeans, x="x", y="y", color="cluster",
                     title="KMeans Clusters UMAP, n_neighbors= " + str(n_neighbors) + ", min_dist= " + str(min_dist),
                     color_discrete_map=color_map)

    fig_kmeans.update_traces(marker=dict(size=5))
    fig_kmeans.show()

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
