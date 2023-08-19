from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np
# Commonly used modules
import numpy as np
# Images, plots, display, and visualization
import matplotlib.pyplot as plt


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

# Convertendo rótulos para representação one-hot
train_labels = to_categorical(train_labels)
test_labels_oneh = to_categorical(test_labels)

# Criando o modelo de Perceptron Logístico
# Este é um classificador logístico, que é uma rede neural de uma única camada com função de ativação sigmoide
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_dim=train_images.shape[1], activation='sigmoid')])

# Definindo o otimizador SGD
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Compilando o modelo
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(train_images, train_labels, epochs=10, batch_size=32, shuffle=True)

# Fazendo previsões
y_pred = model.predict(test_images)
y_pred = np.argmax(y_pred, axis=1)

# Avaliando o modelo
accuracy = accuracy_score(test_labels, y_pred)
print('Accuracy:', accuracy * 100)

# Calculando o total de erros positivos
positive_errors = np.sum((test_labels == 1) & (y_pred != 1))

# Calculando o total de previsões corretas
correct_predictions = np.sum(test_labels == y_pred)

# Calculando o total de verdadeiros positivos
true_positives = np.sum((test_labels == 1) & (y_pred == 1))

# Imprimindo os resultados
print('Número de erros positivos:', positive_errors)
print('Total de previsões corretas:', correct_predictions)
print('Verdadeiros positivos:', true_positives)



