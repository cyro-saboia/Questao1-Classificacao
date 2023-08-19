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

# Função de ativação logística (sigmoide)
def sigmoid(x):
    # Limitar os valores de x para evitar o estouro
    x = np.clip(x, -500, 500)    
    return 1 / (1 + np.exp(-x))

# Função para treinar o Perceptron Logístico
def train_perceptron(X, y, learning_rate=0.1, num_epochs = 50):
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))
    weights = np.zeros((num_classes, num_features))

    for epoch in range(num_epochs):
        for i in range(num_samples):
            # Computar o output do Perceptron
            z = np.dot(weights, X[i])
            y_pred = sigmoid(z)

            # Calcular o erro
            error = y_pred - (y[i] == np.unique(y))

            # Atualizar os pesos
            weights -= learning_rate * np.outer(error, X[i])

    return weights

# Função para realizar a classificação usando o Perceptron Logístico
def predict_perceptron(X, weights):
    z = np.dot(weights, X.T)
    y_pred = sigmoid(z)
    predicted_labels = np.argmax(y_pred, axis=0)
    return predicted_labels

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

# Treinar o Perceptron Logístico
weights = train_perceptron(train_images, train_labels, learning_rate=0.1, num_epochs = 30)

# Realizar a classificação nos dados de teste
predicted_labels = predict_perceptron(test_images, weights)

# Calcular a acurácia
accuracy = np.mean(predicted_labels == test_labels)

# Imprimir a acurácia
print("Acurácia:", accuracy)

#Resultados:
# 1 época: 0,7719
# 30 épocas: 0,8055
# 50 épocas: 0.8055