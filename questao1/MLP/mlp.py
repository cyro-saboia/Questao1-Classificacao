import numpy as np
import gzip
import struct

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função para carregar o conjunto de dados MNIST
def load_mnist(image_file, label_file):
    with gzip.open(label_file, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(image_file, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# Conjunto de dados de treinamento
train_images, train_labels = load_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

# Conjunto de dados de teste
test_images, test_labels = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

# Número de neurônios na camada oculta
hidden_neurons = 1

# Inicialização dos pesos
input_neurons = train_images.shape[1]
output_neurons = 1

# Pesos da camada oculta
weights_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))

# Pesos da camada de saída
weights_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Taxa de aprendizado
learning_rate = 0.1

# Treinamento do modelo
for epoch in range(1000):
    # Forward pass - Camada oculta
    hidden_layer_input = np.dot(train_images, weights_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Forward pass - Camada de saída
    output_layer_input = np.dot(hidden_layer_output, weights_output)
    output_layer_output = sigmoid(output_layer_input)

    # Cálculo do erro
    error = train_labels.reshape(-1, 1) - output_layer_output

    # Backpropagation - Camada de saída
    output_delta = error * output_layer_output * (1 - output_layer_output)
    weights_output += learning_rate * np.dot(hidden_layer_output.T, output_delta)

    # Backpropagation - Camada oculta
    hidden_delta = np.dot(output_delta, weights_output.T) * hidden_layer_output * (1 - hidden_layer_output)
    weights_hidden += learning_rate * np.dot(train_images.T, hidden_delta)

# Teste do modelo
hidden_layer_input = np.dot(test_images, weights_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_output)
output_layer_output = sigmoid(output_layer_input)

# Arredondar os valores de saída para obter as classes previstas
predicted_labels = np.round(output_layer_output)

# Cálculo da acurácia
accuracy = np.mean(predicted_labels == test_labels.reshape(-1, 1)) * 100
print(f"Acurácia do modelo: {accuracy}%")
