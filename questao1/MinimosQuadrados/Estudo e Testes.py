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

test_labels_acuracia = test_labels

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
print('Ypred_q.shape', Ypred_q.shape) # Cotém a posição do array onehot como predição para cada um das 10000 imagens

#----------------------------------------------------------------
# VALIDAÇÃO MANUAL DA PREDIÇÃO EM CIMA DOS LABELS DE TESTE
# print(' ')
# print('Posição',Ypred_q[12]) # Posição no array onehot relevando o label
# print('max',test_labels[:,12])# Validando a Posição no array onehot relevando o label

#----------------------------------------------------------------
# VERIFICAÇÃO/VALIDAÇÃO DOS LABELS DE TESTES E OS PREDITOS
# test_labels = np.argmax(test_labels, axis=0)
# print(' ')
# print('test_labels',test_labels[:12])
# print('Ypred_q',Ypred_q[:12])

# Resultados = np.vstack((test_labels[:12], Ypred_q[:12]))  # Saida desejada e predita lado-a-lado em linha
# print(' ')
# print('Resultados.shape', Resultados.shape)
# print('Resultados.shape.T', Resultados.T.shape)
# print(' ')
# print('Resultados', Resultados[:12])
# print('Resultados.T', Resultados.T[:12])

#----------------------------------------------------------------
# TESTES E VALIDAÇÃODOS RESULTADOS

# Resultados = np.vstack((test_labels, Ypred_q))  # Saida desejada e predita lado-a-lado em linha
# Resultados = Resultados.T

# Erros = Resultados[:12, 0] - Resultados[:12, 1]  # Coluna 1 - Coluna 2
# print(' ')
# print('Erros', Erros)

#----------------------------------------------------------------
# PREPARAÇÃO DOS RESULTADOS

# # Taxas de acerto/erro
Resultados = np.vstack((np.argmax(test_labels, axis=0), Ypred_q))  # Saida desejada e predita lado-a-lado
Resultados = Resultados.T  # Transpor a matriz Resultados
Erros = Resultados[:, 0] - Resultados[:, 1]  # Coluna 1 - Coluna 2

Nerros_pos = len(np.where(Erros > 0)[0])
Nerros_neg = len(np.where(Erros < 0)[0])
Nacertos = test_images.shape[1] - (Nerros_pos + Nerros_neg)

Perros_pos = 100 * Nerros_pos / test_images.shape[1]
Perros_neg = 100 * Nerros_neg / test_images.shape[1]
Pacertos = 100 * Nacertos / test_images.shape[1]

print("Nerros_pos:", Nerros_pos)
print("Nerros_neg:", Nerros_neg)
print("Nacertos:", Nacertos)
print("Perros_pos:", Perros_pos)
print("Perros_neg:", Perros_neg)
print("Pacertos:", Pacertos)


# Calcular a acurácia
accuracy = np.mean(Ypred_q == test_labels_acuracia)

# Imprimir a acurácia
print("Acurácia:", accuracy)