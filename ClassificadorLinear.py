import numpy as np
import gzip, os, hashlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

train_image_file = 'train-images-idx3-ubyte.gz'
train_label_file = 'train-labels-idx1-ubyte.gz'
train_label_file = 't10k-images-idx3-ubyte.gz'
test_label_file = 't10k-labels-idx1-ubyte.gz'

with open(train_image_file, "rb") as f:
    X_train = f.read()

with open(train_label_file, "rb") as f:
    y_train = f.read()

with open(train_label_file, "rb") as f:
    X_test = f.read()

with open(test_label_file, "rb") as f:
    y_test = f.read()        

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