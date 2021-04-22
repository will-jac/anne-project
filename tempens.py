
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from keras import backend as K


from nn_base import build_nn, pretrain_dae_nn
# from util import LabeledData

rng = np.random.default_rng()

UNLABELED = None

# def tempens_loss(unlabeled=-1, num_classes=2, alpha=0.1):
#     '''
#     Custom loss function for temporal ensembling

#     # Needs prior output.
    
#     # Computes MSE between prior output and current output

#     # Computes cross entropy between output and labeled output

#     # Perform a weighted sum of these outputs.

#     '''
#     # def loss(y_true, y_pred):
#     #     # tf.print('pl_loss:', y_true, y_pred)

#     #     y_pl = K.one_hot(K.argmax(y_pred, axis=1), num_classes)
#     #     y_pl = tf.cast(y_pl, y_true.dtype)

#     #     index = y_true == unlabeled
        
#     #     y_pl = tf.where(index, y_pl, y_true)

#     #     index = K.all(index, axis=1)
#     #     coef_arr = tf.where(index, alpha, 1.0)
#     #     # tf.print('coef_arr:', coef_arr)
#     #     # tf.print('labeled:', y_pl[labeled_index], y_pred[labeled_index])
#     #     # tf.print('unlabeled:', y_pl[unlabeled_index], y_pred[unlabeled_index])

#     #     # compute the loss labeled and pl seperately so we can apply alpha
#     #     loss = keras.losses.binary_crossentropy(y_pl, y_pred)
        
#     #     # tf.print('loss, coef_loss:', loss, coef_arr * loss)
#     #     # tf.print('normal loss:', keras.losses.binary_crossentropy(y_true, y_pred))
        
#     #     # loss = keras.losses.categorical_crossentropy(y_pl, y_pred)
        
#     #     return coef_arr * loss

#     return loss

# class TemporalEnsembling():
#     '''
#     Temporal Ensembling is a semi-supervised algorithm for ANN. It works as follows:

#     * train the network with L + U data
#         * For all data perform normal backprop, but customize loss calculation
#         * for labeled data, calculate cross entropy between true and predicted output
#         * for all data, compute MSE between a running recency-weighted average of previous outputs and latest prediction
#         * Add these two loss values together in a weighted sum, MSE gets higher weight longer we run.
#         * Loss is calculated through this special weighted sum. 
#         * Takes three inputs, y_i, z_i, z^bar_i for label, prediction, and prediction record. 
#         * gives two outputs, loss and new prediction record (to be used next time)
    
#     Optionally, this method relies upon dropout and input augmentation

#     This algorithm should hopefully be primarily implemented as a custom loss function with some extra data storage.
#     '''

#     def __init__(self, in_size, hidden, out_size,
#             lrate=0.01, activation='sigmoid', out_activation=None,
#             dropout=0.5, use_dae=True):
#         # store params
#         self.lrate = lrate
#         self.hidden = hidden
#         self.dropout = dropout
#         self.use_dae = use_dae
#         self.activation = activation
#         self.out_activation = out_activation

#         self.callbacks = [
#             keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True, min_delta=0.01)
#         ]

#     def prelabel_data(self, data):
#         label_shape = data.Labeled.y.shape
#         pseudo_labels = -1 * np.ones(label_shape)
#         pseudo_labeled_data = LabeledData(data.Unlabeled.X, pseudo_labels)

#         return LabeledData(
#             np.append(data.Labeled.X, pseudo_labeled_data.X, axis=0),
#             np.append(data.Labeled.y, pseudo_labeled_data.y, axis=0)
#         )

#     def fit(self, train_data, validation_data):

#         train_data = self.prelabel_data(train_data)

#         print('prelabeled data:', train_data)

#         if self.use_dae: # Denoising Autoencoders

#             dae_train_true = np.append(train_data.X, train_data.X, axis=0)
#             dae_valid_true = validation_data.X
#             # add corruption via a bitmask
#             # could also just use a dropout layer here on the input
#             train_mask = rng.integers(low=0, high=1, size=dae_train_true.shape, endpoint=True)
#             valid_mask = rng.integers(low=0, high=1, size=dae_valid_true.shape, endpoint=True)
#             dae_train_corrupted = dae_train_true * train_mask
#             dae_valid_corrupted = dae_valid_true * valid_mask

#             dae_train = LabeledData(dae_train_corrupted, dae_train_true)
#             dae_valid = LabeledData(dae_valid_corrupted, dae_valid_true)

#             # Not using the above right now
#             # dae_train = LabeledData(train_data.Labele
#             # dae_valid = LabeledData(validation_data.X, validation_data.X) 

#             print('building model with dae train', dae_train.X.shape, dae_train.y.shape)

#             self.model = pretrain_dae_nn(dae_train, dae_valid, self.hidden, train_data.Labeled.y.shape[1],
#                     epochs=1, callbacks=self.callbacks,
#                     loss=pl_loss(),
#                     lrate=self.lrate, activation=self.activation, out_activation=self.out_activation,
#                     verbose=True)
#         else:
#             self.model = build_nn(train_data.X.shape[1], self.hidden, train_data.y.shape[1],
#                     lrate=self.lrate, activation=self.activation, out_activation=self.out_activation,
#                     dropout=self.dropout,
#                     loss=pl_loss())

#         print('final model:')
#         self.model.summary()

#         # now, do the training (pseudo-labels & conditional cross-entropy)

#         history = self.model.fit(train_data.X, train_data.y, validation_data=validation_data)

#         print('finished fitting')
#         return history.history

#     def predict(self, X):
#         return self.model.predict(X)

# if __name__ == '__main__':
#     from util import Data, LabeledData, UnlabeledData

#     # small test
#     xor = Data(
#         LabeledData(
#             np.array([
#                 [0,0,0],
#                 [0,0,1], 
#                 # [0,1,0], 
#                 # [0,1,1],
#             ]),
#             np.array([
#                 [1,0], [0,1] #, [1,0], [0,1]
#             ])
#         ), UnlabeledData(
#             np.array([
#                 [1,0,0],
#                 [1,0,1],
#                 # [1,1,0],
#                 # [1,1,1]
#             ])
#         )
#     )

#     model = PseudoLabels(3, [5, 4, 3], 2, use_dae=False)

#     model.fit(xor, xor.Labeled)

import tensorflow as tf
from tensorflow import keras
import numpy as np


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

if __name__ == '__main__':

    # Construct and compile an instance of CustomModel
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Just use `fit` as usual
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    target = np.random.random((1000, 1))
    model.fit(x, y, epochs=3)

