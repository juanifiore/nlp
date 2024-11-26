import os
from sklearn.model_selection import train_test_split
import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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

## RESUMEN DEL CODIGO
#df = pd.read_csv('datasets/topic_sentences.csv')
#df_train, df_test = train_test_split(df, test_size = 0.20, random_state=42)
#docs_train = df_train['text'].tolist()
#docs_test = df_test['text'].tolist()
#
#model = SentenceTransformer("dunzhang/stella_en_400M_v5")
#embeddings_train = model.encode(docs_train)
#embeddings_test = model.encode(docs_test)
#labels_train = df_train['label'].tolist()
#labels_test = df_test['label'].tolist()
#
#rf = RandomForestClassifier(n_estimators=250, max_depth=30, random_state=42)
#rf.fit(embeddings_train, labels_train)
#
#predictions = rf.predict(embeddings_test)
#importancia_dimensiones = rf.feature_importance_



# ===============================================
# CARGAR DATASET Y CREAR LISTA 'docs' 
# ===============================================

print('CARGANDO DATASET....')

# ------------ COMPUTACION DATASET
#df_train = pd.read_csv('datasets/computacion_train.csv')
#df_test = pd.read_csv('datasets/computacion_test.csv')
## achicar datasets
#df_train = df_train.iloc[:1000]
#df_test = df_test.iloc[:500]

# --------------- AG NEWS DATASET
#df = pd.read_csv('datasets/6topic_sentences.csv')
#df_train, df_test = train_test_split(df, test_size = 0.20, random_state=42)
df_train = pd.read_csv('6topic_train.csv')
df_test = pd.read_csv('6topic_test.csv')
#
##armar listas de strings, a los cuales se les calculara los embedding
docs_train = df_train['text'].tolist()
docs_test = df_test['text'].tolist()

#
#
## =====================
## STELLA EMBEDDING MODEL
## ====================
print('CARGANDO MODELO STELLA 400M')
#model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
model = SentenceTransformer(
     "dunzhang/stella_en_400M_v5",
     trust_remote_code=True,
     device="cpu",
     config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
 )
print('CALCULANDO EMBEDDINGS....')
#embeddings_train = model.encode(docs_train)
embeddings_test = model.encode(docs_test)
print('EMBEDDINGS CALCULADOS')

# guardar embeddings calculados 
#np.save('6topic_train.npy',embeddings_train)
np.save('6topic_test.npy',embeddings_test)

embeddings_train = np.load('6topic_train.npy')
#embeddings_test = np.load('embeddings_topics_test.npy')

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


#=========================================== 
# armar lista de labels
labels_train = df_train['label'].tolist()
labels_test = df_test['label'].tolist()
# Convertir las listas a arrays numpy
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
# guardar labels
np.save('labels_6topic_train.npy',labels_train)
np.save('labels_ag_6topic_test.npy',labels_test)


# armar lista de labels
labels_str_train = df_train['label_str'].tolist()
labels_str_test = df_test['label_str'].tolist()
# Convertir las listas a arrays numpy
labels_str_train = np.array(labels_str_train)
labels_str_test = np.array(labels_str_test)
# guardar labels_str
np.save('labels_str_6topic_test.npy',labels_str_train)
np.save('labels_str_6topic_test.npy',labels_str_test)
#=========================================== 

#===========================================
# REDUCIR DIMENCION DE EMBEDDINGS QUEDANDONOS LAS MAS SIGNIFICATIVAS
#dimensiones_mas_relevantes = np.load('embeddings/dim_mas_relevantes_ag_news.npy')
#embeddings_train = embeddings_train[:,dimensiones_mas_relevantes[:25]]
#embeddings_test = embeddings_test[:,dimensiones_mas_relevantes[:25]]
#===========================================


#=========================================== 
# Crear y entrenar el modelo
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    #max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)

# ---------------------------------------------

#rf = RandomForestClassifier(n_estimators=250, max_depth=30, random_state=42)
rf.fit(embeddings_train, labels_train)
#=========================================== 

#=========================================== 
# Realizar predicciones
predictions = rf.predict(embeddings_test)
#=========================================== 

#=========================================== 
# Evaluar el modelo
accuracy = accuracy_score(labels_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(labels_test, predictions))
#=========================================== 

#=========================================== 
# Graficar la importancia de características (opcional)
import matplotlib.pyplot as plt
importances = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.xlabel("Dimensión del Embedding")
plt.ylabel("Importancia")
plt.title("Importancia de las características")
plt.show()
#=========================================== 

