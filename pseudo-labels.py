
import tensorflow as tf
import numpy as np

import tensorflow.keras as keras
from keras import backend as K

from nn_base import build_nn, pretrain_dae_nn

from util import LabeledData

rng = np.random.default_rng()

UNLABELED = None

def pl_loss(unlabeled=-1, num_classes=2, alpha=0.1):
    '''
    Custom loss function for pseudo-labeling

    In general, I believe that this is only well-defined when y is 1D and one-hot encoded

    unlabeled = scalar of what value has been assigned to the unlabled samples
    num_classes = number of classes of Y
    alpha = alpha term of conditional cross-entropy 

    Main limitation: cannot schedule alpha (yet!) - would a callback work?
    '''
    def loss(y_true, y_pred):
        # tf.print('pl_loss:', y_true, y_pred)

        y_pl = K.one_hot(K.argmax(y_pred, axis=1), num_classes)
        y_pl = tf.cast(y_pl, y_true.dtype)

        index = y_true == unlabeled
        
        y_pl = tf.where(index, y_pl, y_true)

        index = K.all(index, axis=1)
        coef_arr = tf.where(index, alpha, 1.0)
        # tf.print('coef_arr:', coef_arr)
        # tf.print('labeled:', y_pl[labeled_index], y_pred[labeled_index])
        # tf.print('unlabeled:', y_pl[unlabeled_index], y_pred[unlabeled_index])

        # compute the loss labeled and pl seperately so we can apply alpha
        loss = keras.losses.binary_crossentropy(y_pl, y_pred)
        
        # tf.print('loss, coef_loss:', loss, coef_arr * loss)
        # tf.print('normal loss:', keras.losses.binary_crossentropy(y_true, y_pred))
        
        # loss = keras.losses.categorical_crossentropy(y_pl, y_pred)
        
        return coef_arr * loss

    return loss

class PseudoLabels():
    '''
    Pseudo Labelling is a semi-supervised algorithm for ANN. It works as follows:

    * train the network with L + U data
        * for labeled data, use normal backprop
        * for unlabeled data, assume the true label is that with the greatest prediction probability
        * This creates a special loss function (the conditional cross entropy)
    * the pseudolabels (assumed true labels) are recalculated with each weight update
    
    Optionally, this method can be improved with dropout and denoising autoencoders for pre-training

    This algorithm is primarily implemented as a custom loss function. Any loss passed in will not be used (yet)
    '''

    def __init__(self, in_size, hidden, out_size,
            lrate=0.01, activation='sigmoid', out_activation=None,
            dropout=0.5, use_dae=True):
        # store params
        self.lrate = lrate
        self.hidden = hidden
        self.dropout = dropout
        self.use_dae = use_dae
        self.activation = activation
        self.out_activation = out_activation

        self.callbacks = [
            keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True, min_delta=0.01)
        ]

    def prelabel_data(self, data):
        label_shape = data.Labeled.y.shape
        pseudo_labels = -1 * np.ones(label_shape)
        pseudo_labeled_data = LabeledData(data.Unlabeled.X, pseudo_labels)

        return LabeledData(
            np.append(data.Labeled.X, pseudo_labeled_data.X, axis=0),
            np.append(data.Labeled.y, pseudo_labeled_data.y, axis=0)
        )

    def fit(self, train_data, validation_data):

        train_data = self.prelabel_data(train_data)

        print('prelabeled data:', train_data)

        if self.use_dae: # Denoising Autoencoders

            dae_train_true = np.append(train_data.X, train_data.X, axis=0)
            dae_valid_true = validation_data.X
            # add corruption via a bitmask
            # could also just use a dropout layer here on the input
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
                    loss=pl_loss(),
                    lrate=self.lrate, activation=self.activation, out_activation=self.out_activation,
                    verbose=True)
        else:
            self.model = build_nn(train_data.X.shape[1], self.hidden, train_data.y.shape[1],
                    lrate=self.lrate, activation=self.activation, out_activation=self.out_activation,
                    dropout=self.dropout,
                    loss=pl_loss())

        print('final model:')
        self.model.summary()

        # now, do the training (pseudo-labels & conditional cross-entropy)

        history = self.model.fit(train_data.X, train_data.y, validation_data=validation_data)

        print('finished fitting')
        return history.history

    def predict(self, X):
        return self.model.predict(X)

if __name__ == '__main__':
    from util import Data, LabeledData, UnlabeledData

    # small test
    xor = Data(
        LabeledData(
            np.array([
                [0,0,0],
                [0,0,1], 
                # [0,1,0], 
                # [0,1,1],
            ]),
            np.array([
                [1,0], [0,1] #, [1,0], [0,1]
            ])
        ), UnlabeledData(
            np.array([
                [1,0,0],
                [1,0,1],
                # [1,1,0],
                # [1,1,1]
            ])
        )
    )

    model = PseudoLabels(3, [5, 4, 3], 2, use_dae=False)

    model.fit(xor, xor.Labeled)



