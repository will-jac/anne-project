
import tensorflow as tf
import numpy as np

import tensorflow.keras as keras

from nn_base import build_nn

X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]

model = build_nn(2, [2, 2], 1, activation='sigmoid', lrate=0.01)

model.summary()

early_stopping_cb = keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True, min_delta=0.01)

history = model.fit(X, y, validation_data=(X, y), epochs=10000, verbose=False, callbacks=[early_stopping_cb])
print(model.predict(X))

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.show()

######################

class PseudoLabels():

    def __init__(self, hidden=[100], lrate=0.01, loss='mse', dropout=0.5, use_dae=True):
        # store params
        self.lrate=lrate
        self.hidden=hidden

    def fit(self, train_data, validation_data):
        self.model = build_nn(train_data[0].shape[0], self.hidden, train_data[1].shape[0])
        history = self.model.fit(X, y, validation_data=validation_data)

        return history.history

    def predict(self, X):
        return self.model.predict(X)
