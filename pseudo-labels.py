
import tensorflow as tf
import numpy as np

import tensorflow.keras as keras

from nn_base import build_nn, pretrain_dae_nn

from util import LabeledData

rng = np.random.default_rng()

class PseudoLabels():

    def __init__(self, in_size, out_size, hidden=[100], lrate=0.01, loss='mse', dropout=0.5, use_dae=True,
            activation='sigmoid', out_activation=None):
        # store params
        self.lrate = lrate
        self.hidden = hidden
        self.loss = loss
        self.dropout = dropout
        self.use_dae = use_dae
        self.activation = activation
        self.out_activation = out_activation

        self.callbacks = [
            keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True, min_delta=0.01)
        ]


    def fit(self, train_data, validation_data):

        if self.use_dae:

            dae_train_true = np.append(train_data.Unlabeled.X, train_data.Labeled.X, axis=0)
            dae_valid_true = validation_data.X
            # add corruption via a bitmask
            train_mask = rng.integers(low=0, high=1, size=dae_train_true.shape, endpoint=True)
            valid_mask = rng.integers(low=0, high=1, size=dae_valid_true.shape, endpoint=True)
            dae_train_corrupted = dae_train_true * train_mask
            dae_valid_corrupted = dae_valid_true * valid_mask

            dae_train = LabeledData(dae_train_corrupted, dae_train_true)
            dae_valid = LabeledData(dae_valid_corrupted, dae_valid_true)

            # Not using the above right now
            # dae_train = LabeledData(train_data.Labele
            # dae_valid = LabeledData(validation_data.X, validation_data.X) 

            print('building model with dae train', dae_train.X.shape, dae_train.y.shape)

            self.model = pretrain_dae_nn(dae_train, dae_valid, self.hidden, train_data.Labeled.y.shape[1],
                    epochs=1, callbacks=self.callbacks,
                    lrate=self.lrate, activation=self.activation, out_activation=self.out_activation,
                    verbose=True)
        else:
            self.model = build_nn(train_data.Labeled.X.shape[1], self.hidden, train_data.Labeled.y.shape[1],
                    lrate=self.lrate, activation=self.activation, out_activation=self.out_activation,
                    dropout=self.dropout)

        print('final model:')
        self.model.summary()

        history = self.model.fit(X, y, validation_data=validation_data)

        return history.history

    def predict(self, X):
        return self.model.predict(X)

if __name__ == '__main__':
    # small test
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    from util import Data, LabeledData, UnlabeledData
    l_data = LabeledData(X, y)
    u_data = UnlabeledData(X)
    data = Data(l_data, u_data)

    model = PseudoLabels(2, 1, [5, 4, 3])

    model.fit(data, l_data)



