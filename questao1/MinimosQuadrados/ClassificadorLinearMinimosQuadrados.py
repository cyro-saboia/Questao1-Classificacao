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

#print(train_labels_onehot[890])
#print(train_labels[890])

# Calcular os pesos do classificador linear de mínimos quadrados
weights = np.linalg.lstsq(train_images, train_labels_onehot, rcond=None)[0]

#print(weights)

# Realizar a classificação nos dados de teste
predictions = test_images @ weights

# Obter os rótulos previstos como o índice do valor máximo em cada linha
predicted_labels = np.argmax(predictions, axis=1)

# Calcular a acurácia
accuracy = np.mean(predicted_labels == test_labels)

# Imprimir a acurácia
print("Acurácia:", accuracy)