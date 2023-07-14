import gzip
import struct
import numpy as np

# Função para carregar o conjunto de dados MNIST
def load_mnist(image_file, label_file):
    with open(label_file, 'rb') as lbpath:
        lbpath.read(8)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(image_file, 'rb') as imgpath:
        imgpath.read(16)
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# LEITURA DOS DADOS
train_images, train_labels = load_mnist('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
test_images, test_labels = load_mnist('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

# Adicionar uma coluna de 1s para o termo de viés
train_images = np.concatenate((train_images, np.ones((train_images.shape[0], 1))), axis=1)
test_images = np.concatenate((test_images, np.ones((test_images.shape[0], 1))), axis=1)

# Codificar os rótulos em one-hot
num_classes = 10
train_labels_onehot = np.eye(num_classes)[train_labels]
test_labels_onehot = np.eye(num_classes)[test_labels]

# Função para centralizar os dados
def center_data(X):
    X_centered = X - np.mean(X, axis=0)
    return X_centered

# Função para normalizar as características dos dados
def normalize_data(X):
    X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)
    return X_normalized

# Função para calcular o PCA
def pca(X, num_components):
    # Centralizar e normalizar os dados
    X_centered = center_data(X)
    X_normalized = normalize_data(X_centered)

    # Calcular a matriz de covariância
    cov_matrix = np.cov(X_normalized, rowvar=False)

    # Calcular os autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Ordenar os autovalores em ordem decrescente
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Selecionar os principais componentes
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Projetar os dados nos principais componentes
    X_projected = np.dot(X_normalized, selected_eigenvectors)

    return X_projected

# Reduzir a dimensionalidade usando o PCA
num_components = 30  # Escolha o número de componentes desejado
train_images_reduced = pca(train_images, num_components)
test_images_reduced = pca(test_images, num_components)

# Calcular os pesos do classificador linear de mínimos quadrados
weights = np.linalg.lstsq(train_images_reduced, train_labels_onehot, rcond=None)[0]

# Realizar a classificação nos dados de teste reduzidos
predictions = test_images_reduced @ weights

# Obter os rótulos previstos como o índice do valor máximo em cada linha
predicted_labels = np.argmax(predictions, axis=1)

# Calcular a acurácia
accuracy = np.mean(predicted_labels == test_labels)

# Imprimir a acurácia
print("Acurácia:", accuracy)