
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