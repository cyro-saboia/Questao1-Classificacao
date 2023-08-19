#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.datasets import mnist


# In[4]:


import tensorflow as tf  # Importando o TensorFlow para carregar o dataset MNIST e para operações de processamento de dados
from sklearn.model_selection import train_test_split  # Importando função para dividir os dados em conjuntos de treino e teste
from sklearn.preprocessing import StandardScaler  # Importando o StandardScaler para normalizar os dados
from sklearn.datasets import fetch_openml  # Importando função para carregar datasets online
from sklearn.metrics import accuracy_score  # Importando a função para calcular a acurácia do modelo
from sklearn.preprocessing import OneHotEncoder  # Importando o OneHotEncoder para transformar as variáveis categóricas em binárias 0 a 9


# In[5]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Commonly used modules
import numpy as np
import os
import sys

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import IPython
from six.moves import urllib

# Pacotes para Regressão
from sklearn.linear_model import LinearRegression

# Pacote para classificação multiclasses
from sklearn.multiclass import OneVsRestClassifier  


print(tf.__version__)


# In[6]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[7]:


print('X_train: ' + str(train_images.shape))
print('Y_train: ' + str(train_labels.shape))
print('X_test:  '  + str(test_images.shape))
print('Y_test:  '  + str(test_labels.shape))


# In[8]:


train_images = train_images.reshape(train_images.shape[0], -1) / 255  # Redimensionando e normalizando as imagens de treino
test_images = test_images.reshape(test_images.shape[0], -1) / 255  # Redimensionando e normalizando as imagens de teste


# In[ ]:





# In[9]:


plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
    #plt.ylabel(i+1)


# In[10]:


regress = LinearRegression()
modelo = OneVsRestClassifier(regress)


# In[11]:


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


# In[ ]:





# In[ ]:




