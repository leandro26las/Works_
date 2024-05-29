import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Definindo o sistema de EDOs
def edo_system(t, y):
    dy1dt = y[1]
    dy2dt = -y[0] + y[2]
    dy3dt = y[3]
    dy4dt = -y[2] + y[0]
    return [dy1dt, dy2dt, dy3dt, dy4dt]

# Gerando dados de treinamento usando o método de Runge-Kutta (solve_ivp)
def generate_data(num_samples, t_span, y0_range):
    X, Y = [], []
    for _ in range(num_samples):
        y0 = np.random.uniform(*y0_range, size=4)
        sol = solve_ivp(edo_system, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 100))
        X.append(y0)
        Y.append(sol.y[:, -1])  # Pega os valores no final do intervalo de tempo
    return np.array(X), np.array(Y)

# Definindo parâmetros e gerando dados
num_samples = 1000
t_span = [0, 10]  # Intervalo de tempo
y0_range = (-1, 1)  # Intervalo para as condições iniciais

X_train, Y_train = generate_data(num_samples, t_span, y0_range)

# Construindo o modelo de rede neural
model = Sequential([
    Dense(64, input_dim=4, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4)
])

# Compilando o modelo
model.compile(optimizer='adam', loss='mse')

# Treinando o modelo
model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2)

# Gerando dados de validação
X_val, Y_val = generate_data(200, t_span, y0_range)

# Avaliando o modelo
loss = model.evaluate(X_val, Y_val)
print(f'Validation loss: {loss}')

# Testando o modelo com novos dados
y0_test = np.random.uniform(*y0_range, size=4)
sol_test = solve_ivp(edo_system, t_span, y0_test, t_eval=np.linspace(t_span[0], t_span[1], 100))
y_pred = model.predict(y0_test.reshape(1, -1))

print("Condições iniciais:", y0_test)
print("Solução numérica no final do intervalo de tempo:", sol_test.y[:, -1])
print("Solução predita pela rede neural:", y_pred[0])
