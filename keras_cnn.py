from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten

from keras.models import Sequential

from data_helpers import load_data_and_labels, load_test

import numpy as np
import data_helpers
import pandas as pd
import os


x, y = load_data_and_labels("train.csv")

# Model
model = Sequential()
model.add(Conv2D(16, (12, 4), padding='valid', input_shape=(14, 4, 1),
                        activation='relu'))

model.add(MaxPooling2D(pool_size=(2,1), padding='valid'))

model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd')

model.summary()

shuffling_index = np.arange(x.shape[0])
np.random.shuffle(shuffling_index)

X_train = x[0:int(x.shape[0]*0.8), :, :]
Y_train = y[0:int(x.shape[0]*0.8)]

X_test = x[int(x.shape[0]*0.8):x.shape[0], :, :]
Y_test = y[int(x.shape[0]*0.8):x.shape[0]]

print(X_train.shape)
print(Y_train.shape)

X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

print(X_train_reshaped.shape)
print(X_test_reshaped.shape)

model.fit(X_train_reshaped, Y_train,
          batch_size=10, epochs=25,
          validation_data=(X_test_reshaped, Y_test))

pred = model.predict(X_test_reshaped, batch_size=32).flatten()
print("Predictions", pred[0:5])

test_x = load_test("test.csv")
test_x_reshaped = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

pred = model.predict(test_x_reshaped, batch_size=32).flatten()
result = np.around(pred).astype(int).tolist()

outputdf = pd.DataFrame(result, columns = ["prediction"])
outputdf.to_csv("result.csv", index_label = "id")