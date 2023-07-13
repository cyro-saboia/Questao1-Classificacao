import numpy as np
import struct

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

print('ORIGINAL')
print('train_images',train_images.shape)
print('train_labels',train_labels.shape)
print('test_images',test_images.shape)
print('test_labels',test_labels.shape)

#----------------------------------------------------------------

# TRANSFORMANDO A CLASSE NUMÉRICA EM ONE-HOT
train_images = train_images.T
train_labels = np.eye(10)[train_labels].T

test_images = test_images.T
test_labels = np.eye(10)[test_labels].T

print(' ')
print('ONEHOT')
print('train_images',train_images.shape)
print('train_labels',train_labels.shape)
print('test_images',test_images.shape)
print('test_labels',test_labels.shape)

#----------------------------------------------------------------

# ALTERANDO A MATRIZ 28X28 DA IMAGEM EM UM VETOR DE 784
train_images = np.vstack((-np.ones((1, train_images.shape[1])), train_images))  # Adiciona uma linha de -1's
test_images = np.vstack((-np.ones((1, test_images.shape[1])), test_images))  # Adiciona uma linha de -1's

print(' ')
print('MATRIX TO ARRAY')
print('train_images',train_images.shape)
print('test_images',test_images.shape)

print(' ')
print('VALIDAÇÃO')
print('train_images',train_images.shape)
print('train_labels',train_labels.shape)
print('train_images.T',train_images.T.shape)

#----------------------------------------------------------------

# CALCULANDO OS PESOS
w = np.dot(train_labels, np.dot(train_images.T, np.linalg.pinv(np.dot(train_images, train_images.T))))

print(' ')
print('MATRIZ DE PESOS')
print('w', w.shape)

#----------------------------------------------------------------

# PREDIÇÃO
Ypred = np.dot(w, test_images)  # Saida como numeros reais
print(' ')
print('PREDIZENDO AS IMAGENS DE TESTES MULTIPLICANDO-AS PELOS PESOS')
np.set_printoptions(threshold=np.inf)
print('Ypred', Ypred.shape)

# RECUPERANDO A CLASSE PREDITA PELO MAIOR VALOR
Ypred_q = np.argmax(Ypred, axis=0) 
print(' ')
print('CLASSE PELO MAIOR VALOR')
print(Ypred_q.shape)

#----------------------------------------------------------------
max = np.argmax(test_labels, axis=0)
print(' ')
print('max',max[198])


print('Ypred_q', Ypred_q.shape)

