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

embeddings_test = np.load('embeddings/embeddings_topics_test.npy')
embeddings_train = np.load('embeddings/embeddings_topics_train.npy')
labels_test = np.load('embeddings/labels_topics_test.npy')
labels_train = np.load('embeddings/labels_topics_train.npy')

#===========================================
# REDUCIR DIMENCION DE EMBEDDINGS QUEDANDONOS LAS MAS SIGNIFICATIVAS
#dimensiones_mas_relevantes = np.load('embeddings/dim_mas_relevantes_ag_news.npy')
dimensiones_mas_relevantes = np.load('embeddings/dim_mas_relevantes_topics.npy')
dimensiones_mas_relevantes = dimensiones_mas_relevantes[::-1] 
#embeddings_train = embeddings_train[:,dimensiones_mas_relevantes[:25]]
#embeddings_test = embeddings_test[:,dimensiones_mas_relevantes[:25]]
#===========================================


#=========================================== 
# Crear y entrenar el modelo
rf = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42)
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
#import matplotlib.pyplot as plt
#importances = rf.feature_importances_
#plt.figure(figsize=(10, 6))
#plt.bar(range(len(importances)), importances)
#plt.xlabel("Dimensión del Embedding")
#plt.ylabel("Importancia")
#plt.title("Importancia de las características")
#plt.show()
#=========================================== 

