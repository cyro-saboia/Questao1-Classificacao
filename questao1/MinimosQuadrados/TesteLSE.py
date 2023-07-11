import gzip
import struct
import numpy as np

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

# Codificar os rótulos em one-hot
num_classes = 10
train_labels_onehot = np.eye(num_classes)[train_labels]
test_labels_onehot = np.eye(num_classes)[test_labels]

print('train_images.shape', train_images.shape)
print('train_labels.shape', train_labels.shape)
print('test_images.shape', test_images.shape)
print('test_labels.shape', test_labels.shape)

print('train_labels_onehot.shape', train_labels_onehot.shape)
print('test_labels_onehot.shape', test_labels_onehot.shape)

y = train_labels_onehot
X = train_images
XT = np.transpose(X)


print('y.shape', y.shape)
print('X.shape',X.shape)
print('XT.shape',XT.shape)

C = X@XT

print('X*XT.shape', C.shape)


w = y * XT # * (linalg.inv(X * XT))
print(w.shape)