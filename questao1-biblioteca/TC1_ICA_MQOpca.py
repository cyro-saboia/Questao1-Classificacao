from keras.datasets import mnist
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA  # Importando PCA
# TensorFlow and tf.keras
import tensorflow as tf
from sklearn.decomposition import PCA  # Importando PCA
# Commonly used modules
import numpy as np
# Images, plots, display, and visualization
import matplotlib.pyplot as plt
# Pacotes para Regressão
from sklearn.linear_model import LinearRegression
# Pacote para classificação multiclasses
from sklearn.multiclass import OneVsRestClassifier  

print(tf.__version__)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


print('X_train: ' + str(train_images.shape))
print('Y_train: ' + str(train_labels.shape))
print('X_test:  '  + str(test_images.shape))
print('Y_test:  '  + str(test_labels.shape))

train_images = train_images.reshape(train_images.shape[0], -1) / 255  # Redimensionando e normalizando as imagens de treino
test_images = test_images.reshape(test_images.shape[0], -1) / 255  # Redimensionando e normalizando as imagens de teste

plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
    #plt.ylabel(i+1)

pca = PCA(n_components=100)
train_images = pca.fit_transform(train_images)
test_images = pca.transform(test_images)

regress = LinearRegression()
modelo = OneVsRestClassifier(regress)


modelo.fit(train_images, train_labels)  # Treinamento do modelo

# Fazendo previsões
pred = modelo.predict(test_images)  # Fazendo previsões nos dados de teste

# Calculando a acurácia
acuracia = accuracy_score(test_labels, pred)  # Comparando as previsões com os valores reais

# Calculando o total de erros positivos
erros_positivos = np.sum((test_labels == 1) & (pred != 1))

# Calculando o total de previsões corretas
acertos = np.sum(test_labels == pred)

# Calculando o total de verdadeiros positivos
acertos_positivos = np.sum((test_labels == 1) & (pred == 1))

# Imprimindo os resultados
print('Acuracia:', acuracia * 100)  # Imprime a acurácia em porcentagem
print('Número de erros positivos:', erros_positivos)  # Imprime o número de erros positivos
print('Total de previsões corretas:', acertos)  # Imprime o total de previsões corretas
print('Verdadeiros positivos:', acertos_positivos)  # Imprime o número de verdadeiros positivos



