import numpy as np
import matplotlib.pyplot as plt

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

# Passo 1: Carregar banco de dados
X = train_images.T
D = train_labels.reshape(-1, 1)

# Passo 2: Obter informações dos dados
n = X.shape  # n[0] = dimensão da entrada, n[1] = número de exemplos
N = n[1]  # Número de exemplos = número de colunas da matriz X

# Passo 3: Estimar os parâmetros do classificador (pesos e limiares)
p = n[0]  # dimensão do vetor de entrada
Ne = 300  # Número de épocas de treinamento
alfa = 0.001  # Taxa de aprendizagem

W = np.random.rand(p, 1)  # Inicialização do vetor de pesos

erro_medio_epoca = np.zeros(Ne)

for t in range(Ne):
    Itr = np.random.permutation(N)
    Xtr = X[:, Itr]  # Embaralha dados a cada época de treinamento
    Dtr = D[Itr]

    acc_erro_quad = 0  # Acumula erro quadrático por vetor em uma época
    for k in range(N):
        ypred = np.sign(np.dot(W.T, Xtr[:, k]))  # Saída predita para o k-ésimo vetor de entrada
        erro = Dtr[k] - ypred  # erro de predição
        W = W + alfa * erro * Xtr[:, k]  # Atualização do vetor de pesos
        acc_erro_quad = acc_erro_quad + 0.5 * erro * erro

    erro_medio_epoca[t] = acc_erro_quad / N

plt.plot(erro_medio_epoca)
plt.title('Curva de Aprendizagem')
plt.xlabel('Época de treinamento')
plt.ylabel('Erro quadrático médio por época')
plt.show()

# Passo 4: Determinar predições da classe dos vetores de teste
Xts = test_images.T
Dts = test_labels.reshape(-1, 1)

Ypred = np.sign(np.dot(W.T, Xts))  # Saída quantizada para +1 ou -1

# Passo 5: Determinar as taxas de acerto/erro
Erros = Dts - Ypred  # Coluna 1 - Coluna 2

Nerros_pos = len(np.where(Erros > 0)[0])
Nerros_neg = len(np.where(Erros < 0)[0])
Nacertos = N - (Nerros_pos + Nerros_neg)

Perros_pos = 100 * Nerros_pos / Dts.shape[0]
Perros_neg = 100 * Nerros_neg / Dts.shape[0]
Pacertos = 100 * Nacertos / Dts.shape[0]

print('Taxa de erros positivos: {:.2f}%'.format(Perros_pos))
print('Taxa de erros negativos: {:.2f}%'.format(Perros_neg))
print('Taxa de acertos: {:.2f}%'.format(Pacertos))
