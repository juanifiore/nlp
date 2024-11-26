import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

embeddings_train = np.load('embeddings/6topic_train.npy')
embeddings_test = np.load('embeddings/6topic_test.npy')

labels_train = np.load('embeddings/labels_6topic_train.npy')
labels_test = np.load('embeddings/labels_6topic_test.npy')

importancia_dimensiones = np.load('embeddings/importancias_6topic.npy')

dimensiones_a_probar = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] 

resultados = []

for num_dim in dimensiones_a_probar:
    print(f"Probando con las {num_dim} dimensiones más importantes...")

    # Seleccionar las 'num_dim' dimensiones más importantes
    indices_dimensiones = importancia_dimensiones[:num_dim]
    embeddings_train_reducido = embeddings_train[:, indices_dimensiones]
    embeddings_test_reducido = embeddings_test[:, indices_dimensiones]

    start_time = time.time()

    # Entrenar el modelo Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced', random_state=42)
    rf.fit(embeddings_train_reducido, labels_train)

    # Medir tiempo de fiteo
    end_time = time.time()
    execution_time = end_time - start_time

    # Evaluar el modelo
    predictions = rf.predict(embeddings_test_reducido)
    accuracy = accuracy_score(labels_test, predictions)
    print(f"Accuracy con {num_dim} dimensiones: {accuracy:.4f}")
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

    # Guardar resultados
    resultados.append({"dimensiones": num_dim, "accuracy": accuracy, 
                       "clasif_rep": classification_report(labels_test, predictions),
                       "matrix": confusion_matrix(labels_test, predictions),
                       "tiempo": execution_time})

# Convertir resultados a un DataFrame para analizar
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('rf_6topic_dimensiones.csv')

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(df_resultados["dimensiones"], df_resultados["accuracy"], marker='o')
plt.xlabel("Cantidad de Dimensiones")
plt.ylabel("Accuracy")
plt.title("Efecto de la cantidad de dimensiones en la Accuracy")
plt.grid()
plt.show()

# Graficar los tiempos 
plt.figure(figsize=(10, 6))
plt.plot(df_resultados["dimensiones"], df_resultados["tiempo"], marker='o')
plt.xlabel("Cantidad de Dimensiones")
plt.ylabel("Tiempo de entrenamiento")
plt.title("Efecto de la cantidad de dimensiones en el tiempo de ejecucion")
plt.grid()
plt.show()
