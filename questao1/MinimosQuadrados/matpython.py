import numpy as np
import gzip
import struct

# Função para carregar o conjunto de dados MNIST
def load_mnist(image_file, label_file):
    with gzip.open(label_file, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(image_file, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

# Carregar o conjunto de dados de treinamento
train_images, train_labels = load_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

# Carregar o conjunto de dados de teste
test_images, test_labels = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')


# Passo 1: Carregar banco de dados
Xtr = train_images.T
Dtr = np.eye(10)[train_labels].T

Xts = test_images.T
Dts = np.eye(10)[test_labels].T

# Passo 3: Estimar os parametros do classificador (pesos e limiares)
Xtr = np.vstack((-np.ones((1, Xtr.shape[1])), Xtr))  # Adiciona uma linha de -1's
Xts = np.vstack((-np.ones((1, Xts.shape[1])), Xts))  # Adiciona uma linha de -1's

W = np.dot(Dtr, np.dot(Xtr.T, np.linalg.pinv(np.dot(Xtr, Xtr.T))))

# Passo 4: Determinar predicoes da classe dos vetores de teste
Ypred = np.dot(W, Xts)  # Saida como numeros reais
Ypred_q = np.argmax(Ypred, axis=0)  # Encontrar a classe predita com maior valor

print(np.argmax(Dts, axis=0)[:10])
print(Ypred_q[:10])

# Passo 5: Determinar as taxas de acerto/erro
Resultados = np.vstack((np.argmax(Dts, axis=0), Ypred_q))  # Saida desejada e predita lado-a-lado
Resultados = Resultados.T  # Transpor a matriz Resultados
Erros = Resultados[:, 0] - Resultados[:, 1]  # Coluna 1 - Coluna 2

Nerros_pos = len(np.where(Erros > 0)[0])
Nerros_neg = len(np.where(Erros < 0)[0])
Nacertos = Xts.shape[1] - (Nerros_pos + Nerros_neg)

Perros_pos = 100 * Nerros_pos / Xts.shape[1]
Perros_neg = 100 * Nerros_neg / Xts.shape[1]
Pacertos = 100 * Nacertos / Xts.shape[1]

print("Nerros_pos:", Nerros_pos)
print("Nerros_neg:", Nerros_neg)
print("Nacertos:", Nacertos)
print("Perros_pos:", Perros_pos)
print("Perros_neg:", Perros_neg)
print("Pacertos:", Pacertos)
