import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar a base de dados MNIST
mnist = fetch_openml('mnist_784', as_frame=False)

# Obter os dados e rótulos
X, y = mnist['data'], mnist['target']

# Pré-processamento dos dados
X = StandardScaler().fit_transform(X.astype(np.float32))
y = y.astype(int)  # Modificação aqui

# Dividir em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converter os dados para tensores do PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Definir a arquitetura do Perceptron Logístico Sigmoidal
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Configurar hiperparâmetros
input_dim = X_train_tensor.shape[1]
output_dim = len(np.unique(y_train))
learning_rate = 0.1
num_epochs = 10000

# Inicializar o modelo
model = LogisticRegression(input_dim, output_dim)

# Definir a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Treinamento do modelo
for epoch in range(num_epochs):
    # Forward pass e cálculo da função de perda
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Avaliação do modelo nos dados de teste
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Acurácia nos dados de teste: {accuracy:.2f}')
