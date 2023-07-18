import numpy as np
import matplotlib.pyplot as plt

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

# DEFINE ARQUITETURA DA REDE
#===========================
Ne = 500  # No. de epocas de treinamento
Nr = 1    # No. de rodadas de treinamento/teste
Nh = 3    # No. de neuronios na camada oculta
No = 1    # No. de neuronios na camada de saida

eta = 0.10  # Passo de aprendizagem
mom = 0.40  # Fator de momento

# Inicio do Treino
for r in range(Nr):  # LOOP DE RODADAS TREINO/TESTE
    Rodada = r + 1

    # Embaralha vetores de entrada e saídas desejadas
    I = np.random.permutation(len(train_labels))
    train_images = train_images[I]
    train_labels = train_labels[I]

    lP, cP = train_images.shape  # Tamanho da matriz de vetores de treinamento
    lQ, cQ = test_images.shape  # Tamanho da matriz de vetores de teste

    # Inicia matrizes de pesos
    WW = 0.1 * np.random.rand(Nh, lP + 1)  # Pesos entrada -> camada oculta
    WW_old = WW.copy()  # Necessário para termo de momento

    MM = 0.1 * np.random.rand(No, Nh + 1)  # Pesos camada oculta -> camada de saída
    MM_old = MM.copy()  # Necessário para termo de momento

    # ETAPA DE TREINAMENTO
    for t in range(Ne):  # Inicia LOOP de épocas de treinamento
        Epoca = t + 1

        I = np.random.permutation(cP)
        # P = P[:, I]
        train_labels = train_labels[I]  # Embaralha vetores de treinamento

        EQ = 0
        HID1 = []
        for tt in range(cP):  # Inicia LOOP de iterações em uma época de treinamento
            # CAMADA OCULTA
            X = np.concatenate(([-1], train_labels[:, tt]))  # Constroi vetor de entrada com adição da entrada x0=-1
            Ui = WW.dot(X)  # Ativação (net) dos neurônios da camada oculta
            Zi = sigmoid(Ui)  # Saída entre [0,1] (função logística)
            HID1.append(Zi)

            # CAMADA DE SAÍDA
            Z = np.concatenate(([-1], Zi))  # Constroi vetor de entrada DESTA CAMADA com adição da entrada y0=-1
            Uk = MM.dot(Z)  # Ativação (net) dos neurônios da camada de saída
            Yk = sigmoid(Uk)  # Saída entre [0,1] (função logística)

            # CÁLCULO DO ERRO
            Ek = train_labels[tt] - Yk  # erro entre a saída desejada e a saída da rede
            EQ += 0.5 * np.sum(Ek**2)  # soma do erro quadrático de todos os neurônios p/ VETOR DE ENTRADA

            # CALCULO DOS GRADIENTES LOCAIS
            Dk = Yk * (1 - Yk) + 0.01  # derivada da sigmoide logística (camada de saída)
            DDk = Ek * Dk  # gradiente local (camada de saída)

            Di = Zi * (1 - Zi) + 0.01  # derivada da sigmoide logística (camada oculta)
            DDi = Di * (MM[:, 1:].T.dot(DDk))  # gradiente local (camada oculta)

            # AJUSTE DOS PESOS - CAMADA DE SAÍDA
            MM_aux = MM.copy()
            MM += eta * DDk.reshape(-1, 1) * Z + mom * (MM - MM_old)
            MM_old = MM_aux

            # AJUSTE DOS PESOS - CAMADA OCULTA
            WW_aux = WW.copy()
            WW += 2 * eta * DDi.reshape(-1, 1) * X + mom * (WW - WW_old)
            WW_old = WW_aux

        EQM = EQ / cP  # MEDIA DO ERRO QUADRATICO POR EPOCA

    plt.figure()
    plt.plot(range(Ne), EQM)
    plt.xlabel('épocas de treinamento')
    plt.ylabel('EQM')
    plt.title('Erro quadrático médio (EQM) por época')

    # ETAPA DE GENERALIZAÇÃO
    EQ2 = 0
    HID2 = []
    OUT2 = []
    for tt in range(cQ):
        # CAMADA OCULTA
        X = np.concatenate(([-1], test_images[:, tt]))  # Constroi vetor de entrada com adição da entrada x0=-1
        Ui = WW.dot(X)  # Ativação (net) dos neurônios da camada oculta
        Zi = sigmoid(Ui)  # Saída entre [0,1] (função logística)
        HID2.append(Zi)

        # CAMADA DE SAÍDA
        Z = np.concatenate(([-1], Zi))  # Constroi vetor de entrada DESTA CAMADA com adição da entrada y0=-1
        Uk = MM.dot(Z)  # Ativação (net) dos neurônios da camada de saída
        Yk = sigmoid(Uk)  # Saída entre [0,1] (função logística)
        OUT2.append(Yk)  # Armazena saída da rede

        # ERRO QUADRÁTICO GLOBAL (todos os neurônios) POR VETOR DE ENTRADA
        Ek = test_labels[tt] - Yk
        EQ2 += 0.5 * np.sum(Ek**2)

    # MEDIA DO ERRO QUADRATICO COM REDE TREINADA (USANDO DADOS DE TESTE)
    EQM2 = EQ2 / cQ

    # CALCULA TAXA DE ACERTO
    count_OK = 0  # Contador de acertos
    for tt in range(cQ):
        iT2max = np.argmax(test_labels[tt])  # Índice da saída desejada de maior valor
        iOUT2max = np.argmax(OUT2[tt])  # Índice do neurônio cuja saída é a maior
        if iT2max == iOUT2max:  # Conta acerto se os dois índices coincidem
            count_OK += 1

    # Taxa de acerto global
    Tx_OK = 100 * (count_OK / cQ)
    print(f"Taxa de acerto da rodada {Rodada}: {Tx_OK}%")

# Taxa média de acerto global
Tx_media = np.mean(Tx_OK)
Tx_std = np.std(Tx_OK)

print(f"Taxa média de acerto global: {Tx_media}%")
print(f"Desvio padrão da taxa média de acerto: {Tx_std}%")

# Plot SUPERFÍCIE DE DECISÃO
incr = 0.05
Xmin, Xmax = np.min(train_images[:, 0]), np.max(train_images[:, 0])
Ymin, Ymax = np.min(train_images[:, 1]), np.max(train_images[:, 1])

Lx = np.arange(Xmin, Xmax, incr)
Ly = np.arange(Ymin, Ymax, incr)
LX, LY = np.meshgrid(Lx, Ly)
SD1 = []

for i in range(len(Lx)):
    for j in range(len(Ly)):
        X = np.concatenate(([-1], [Lx[i]], [Ly[j]]))
        Ui = WW.dot(X)
        Zi = sigmoid(Ui)
        Z = np.concatenate(([-1], Zi))
        Uk = MM.dot(Z)
        Yk = sigmoid(Uk)

        if np.round(Yk) == 1:
            SD1.append([Lx[i], Ly[j], 1, 0, 0])
        else:
            SD1.append([Lx[i], Ly[j], 0, 1, 0])

SD1 = np.array(SD1)

fig, ax = plt.subplots()
ax.scatter(SD1[:, 0], SD1[:, 1], c=SD1[:, 2:5], cmap='coolwarm')
ax.plot(train_images[train_labels == 0, 0], train_images[train_labels == 0, 1], 'ro')
ax.plot(train_images[train_labels == 1, 0], train_images[train_labels == 1, 1], 'bo')
ax.set_xlim(Xmin, Xmax)
ax.set_ylim(Ymin, Ymax)
plt.show()
