import yfinance as yf
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# отримуємо дані з  Yahoo Finance

stock_name = 'TSLA'
data = yf.download(stock_name, start="2020-03-26", end="2021-03-29")


# обираємо ціну Close акції
data = data.filter(['Close'])
# конвертуємо дані у масив
dataset = data.values
# кількість даних для тренування моделі - 80%, тест - 20%
training_data_len = int(np.ceil( len(dataset) * .80 ))



# масштабування даних відносно відрізку від 0 до 1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)



# створюємо масштабований масив даних
train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))



from keras.models import Sequential
from keras.layers import Dense, LSTM

# Будуємо LSTM модель
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.35))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(1))
# Компіляція
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Тренування
model.fit(x_train, y_train, batch_size=1, epochs=21)


# Малюємо структуру моделі
keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)



# Створюємо сет даних для тестування моделі
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Передбачення на основі тестових даних
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Знаходимо похибку
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse)


# Малюємо графік
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()



# Робимо передбачення на наступний місяць
data_new = yf.download(stock_name, start="2021-03-01", end="2021-04-30")

data_new = data_new.filter(['Close'])
dataset = data_new.values
training_data_len = len(dataset)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

test_data = scaled_data[training_data_len - len(data_new): , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(20, len(test_data)):
    x_test.append(test_data[i-20:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

hist_data_new = yf.download(stock_name, start="2021-04-01", end="2021-05-04")
hist_data_new = hist_data_new.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
hist_data_new = hist_data_new['Close']
hist_data_new = np.array(hist_data_new)
pred_lstm = model.predict(x_test)
pred_lstm = scaler.inverse_transform(pred_lstm)
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.ylabel('Prices')
plt.plot(pred_lstm, label = 'predicted')
plt.plot(hist_data_new, label = 'historical')
plt.title(f'{stock_name} predicted price')
plt.legend()
plt.show()





