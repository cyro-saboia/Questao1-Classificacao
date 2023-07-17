import numpy as np
from sklearn.decomposition import PCA

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

# Aplicar PCA aos dados de treinamento
pca = PCA(n_components=30)  # Defina o número de componentes principais desejados
train_images_pca = pca.fit_transform(train_images)

# Aplicar PCA aos dados de teste
test_images_pca = pca.transform(test_images)

# np.set_printoptions(threshold=np.inf)
# print(train_images_pca[:1])
# print(test_images_pca[:1])

Xtr = train_images_pca.T
Dtr = np.eye(10)[train_labels].T

Xts = test_images_pca.T
Dts = np.eye(10)[test_labels].T

# Parametros do classificador
Xtr = np.vstack((-np.ones((1, Xtr.shape[1])), Xtr))  # Adiciona uma linha de -1's
Xts = np.vstack((-np.ones((1, Xts.shape[1])), Xts))  # Adiciona uma linha de -1's

# Calcular pesos
W = np.dot(Dtr, np.dot(Xtr.T, np.linalg.pinv(np.dot(Xtr, Xtr.T))))

# Predição
Ypred = np.dot(W, Xts)  # Saida como numeros reais
Ypred_q = np.argmax(Ypred, axis=0)  # Encontrar a classe predita com maior valor

# Taxas de acerto/erro
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

# Calcular a acurácia
accuracy = np.mean(Ypred_q == test_labels)

# Imprimir a acurácia
print("Acurácia:", accuracy)
