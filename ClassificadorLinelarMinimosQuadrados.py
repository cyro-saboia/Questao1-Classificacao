import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados MNIST
mnist = fetch_openml('mnist_784')

# Separar os dados de treinamento e teste
X_train = mnist.data[:60000]
y_train = mnist.target[:60000]
X_test = mnist.data[60000:]
y_test = mnist.target[60000:]

# Criar o classificador linear
clf = LinearRegression()

# Treinar o classificador
clf.fit(X_train, y_train)

# Realizar a classificação nos dados de teste
y_pred = clf.predict(X_test)

# Arredondar os valores previstos para os rótulos de classe
y_pred = np.round(y_pred)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)

# Imprimir a acurácia
print("Acurácia:", accuracy)