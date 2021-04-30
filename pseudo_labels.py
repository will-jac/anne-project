
import tensorflow as tf
import numpy as np

import tensorflow.keras as keras
from tensorflow.keras import backend as K

from nn_base import build_nn, pretrain_dae_nn

from util import Data

from generator import pseudo_label_generator

rng = np.random.default_rng()

UNLABELED = None

# TODO: schedule alpha

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

    def __init__(self, model, 
            lrate=0.0001, 
            epochs=100, 
            batch_size=100,
            steps_per_epoch=2,
            patience=500, min_delta=0.0,
            use_image_augmentation=False,
            use_dae=False, pretrain_lrate=0.001, pretrain_epochs=100,
            af = 3.0, T1 = 200, T2 = 800,
            out_size = 10
        ):
        # store params
        self.lrate = lrate
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.steps_per_epoch=steps_per_epoch
        self.use_dae = use_dae
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lrate = pretrain_lrate

        self.callbacks = [
            keras.callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True, min_delta=min_delta,
                monitor='categorical_accuracy'
            ),
            # alpha scheduler
            keras.callbacks.LambdaCallback(
                on_epoch_begin = lambda epoch, logs : self.alpha_schedule(epoch)
            )
        ]

        self.metrics = [
            keras.metrics.CategoricalAccuracy(),
        ]

        self.name='pseudo_labels'

        self.unlabeled = -1.0

        self.model = model(
            output_activation='softmax',
            do_image_augmentation = use_image_augmentation
        )

        ## for scheduling alpha
        self.af = af
        self.T1 = T1
        self.T2 = T2

        self.alpha = 0.0 # initially

        self.out_size = out_size

    def alpha_schedule(self, epoch_num):
        if epoch_num < self.T1:
            print('alpha set to 0.0')
            return 0.0
        if epoch_num < self.T2:
            print('alpha set to', (epoch_num - self.T1) / (self.T2 - self.T1) * self.af)
            return (epoch_num - self.T1) / (self.T2 - self.T1) * self.af
        else:
            print('alpha set to', self.af)
            return self.af

    def pl_loss(self, y_true, y_pred):
        '''
        Custom loss function for pseudo-labeling

        In general, I believe that this is only well-defined when y is 1D and one-hot encoded

        unlabeled = scalar of what value has been assigned to the unlabled samples
        num_classes = number of classes of Y
        alpha = alpha term of conditional cross-entropy
        '''
        # tf.print('pl_loss:', y_true, y_pred)

        y_pl = K.one_hot(K.argmax(y_pred, axis=1), self.out_size)
        y_pl = tf.cast(y_pred, y_true.dtype)

        # Calculate whether each sample in the batch is labeled or unlabeled
        index = y_true == self.unlabeled

        # clip the predictions for numerical stability
        # pred = K.clip(y_pred, 1e-12, 1 - 1e-12)
        
        # tf.print('y_pred', y_pred.shape, y_pred)
        # tf.print('y_true', y_true.shape, y_true)
        # tf.print('index', index.shape, index)
        
        y_pl = tf.where(index, y_pl, y_true)
        # tf.print('y_pl', y_pl.shape, y_pl)
        
        # tf.print('index', index.shape, index)
        # Set coefficient for each sample based on whether labeled or unlabeled
        index = K.all(index, axis=1)
        coef_arr = tf.where(index, self.alpha, 1.0)
        tf.print('coef_arr', coef_arr.shape)

        # compute the loss
        loss = keras.losses.categorical_crossentropy(y_pl, y_pred)
        tf.print('loss', loss.shape)
        # loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

        return coef_arr * loss

    def prelabel_data(self, data):
        if data.y.shape[1] is None:
            label_shape = tuple(data.U.shape[0])
        else:
            label_shape = (data.U.shape[0], data.y.shape[1])
        pseudo_labels = self.unlabeled * np.ones(label_shape, dtype=float)

        X = np.append(data.X, data.U, axis=0)
        y = np.append(data.y, pseudo_labels, axis=0)

        

        return Data(
            np.append(data.X, data.U, axis=0),
            np.append(data.y, pseudo_labels, axis=0),
            None
        )

    def fit(self, train_data, validation_data):
        print('train data:', train_data.X.shape, train_data.y.shape, train_data.U.shape)
        u = train_data.U.shape[0]

        train_data = self.prelabel_data(train_data)

        # print('prelabeled data:', train_data.X.shape, train_data.y.shape)

        if self.use_dae:

            dae_train_true = np.append(train_data.X, train_data.X, axis=0)
            dae_valid_true = validation_data.X
            # add corruption via a bitmask
            # could also just use a dropout layer here on the input
            train_mask = rng.integers(low=0, high=1, size=dae_train_true.shape, endpoint=True)
            valid_mask = rng.integers(low=0, high=1, size=dae_valid_true.shape, endpoint=True)
            dae_train_corrupted = dae_train_true * train_mask
            dae_valid_corrupted = dae_valid_true * valid_mask

            dae_train = Data(dae_train_corrupted, dae_train_true, None)
            dae_valid = Data(dae_valid_corrupted, dae_valid_true, None)

            # Not using the above right now
            # dae_train = LabeledData(train_data.Labele
            # dae_valid = LabeledData(validation_data.X, validation_data.X)

            print('building model with dae train', dae_train.X.shape, dae_train.y.shape)

            self.model.pretrain(dae_train, dae_valid, self.pretrain_epochs, self.pretrain_lrate)

        print('final model:')
        self.model.summary()

        # compile the model
        self.model.compile(
            loss=self.pl_loss, 
            optimizer=tf.keras.optimizers.Adam(lr=self.lrate),
            metrics=self.metrics
        )

        # now, do the training (pseudo-labels & conditional cross-entropy)

        print('training on:', train_data.X.shape, train_data.y.shape)
        tf.keras.Model().fit()
        history = self.model.fit(
            np.array(train_data.X), 
            batch_size=self.batch_size,
            validation_data=(np.array(validation_data.X), np.array(validation_data.y)),
            callbacks=self.callbacks,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch
        )
        print('finished fitting')
        return history.history

    def predict(self, X):
        return self.model.predict(X)

# if __name__ == '__main__':
#     from util import Data

#     # small test
#     xor = Data(
#         np.array([
#             [0,0,0],
#             [0,0,1],
#             # [0,1,0],
#             # [0,1,1],
#         ]),
#         np.array([
#             [1,0], [0,1] #, [1,0], [0,1]
#         ]),
#         np.array([
#             [1,0,0],
#             [1,0,1],
#             # [1,1,0],
#             # [1,1,1]
#         ])
#     )

#     model = PseudoLabels(3, [5, 4, 3], 2, use_dae=False)

#     model.fit(xor, xor.Labeled)



