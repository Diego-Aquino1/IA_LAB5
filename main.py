import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def kmeans(X, k, max_iterations=100):
    # Inicialización aleatoria de centroides
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]
    
    for _ in range(max_iterations):
        # Calcular distancias entre puntos y centroides
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # Encontrar la etiqueta de clúster más cercana para cada punto
        labels = np.argmin(distances, axis=1)
        
        # Actualizar posiciones de centroides
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Mostrar tabla de distancias en cada iteración
        print("Iteración:", _ + 1)
        print("Punto  | X    | Y    | Distancia C1 | Distancia C2 | Distancia C3 | Distancia mínima")
        print("-----------------------------------------------")
        for i, (x, y) in enumerate(X):
            dist_to_centroids = distances[i]
            min_distance = dist_to_centroids[labels[i]]
            print(f"{i+1:5}  | {x:.2f} | {y:.2f} | {dist_to_centroids[0]:.4f}    | {dist_to_centroids[1]:.4f}    | {dist_to_centroids[2]:.4f}    | {min_distance:.4f}")
        print("-----------------------------------------------")
        
        # Verificar si los centroides se han estabilizado
        if np.all(centroids == new_centroids):
            break 
        
        centroids = new_centroids
    
    return centroids, labels

# Generar datos de muestra
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Ejecutar el algoritmo K-means
k = 3
centroids, labels = kmeans(X, k)

# Mostrar resultados
print("Centroides finales:")
print(centroids)

# Mostrar gráfico con los puntos y centroides
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.show()