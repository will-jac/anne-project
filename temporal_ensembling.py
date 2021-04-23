# from scratch tensorflow implementation of temporal ensembling 
#   since I couldn't even get theano / lasagne to import, in part due to the discontinuation
#   of development on theano, I thought this would be a more principled approach.

import math
import numpy as np 
import tensorflow as tf

# https://www.tensorflow.org/addons/tutorials/layers_weightnormalization
# import tensorflow_addons as tfa
from weight_normalization import WeightNormalization

import tensorflow_addons as tfa

# for more information about how to use the loss / gradients defined here, see
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

def temporal_ensembling_loss(X, y, U, model, unsupervised_weight, ensembling_targets):
    z_X = model(X)
    z_U = model(U)

    pred = tf.concat([z_X, z_U], 0)

    return pred, tf.compat.v1.losses.softmax_cross_entropy(y, z_X) + \
            unsupervised_weight * (tf.keras.losses.mean_squared_error(ensembling_targets, pred))

def temporal_ensembling_gradients(X, y, U, model, unsupervised_weight, ensembling_targets):

    with tf.GradientTape() as tape:
        ensemble_precitions, loss_value = temporal_ensembling_loss( 
            X, y, U,
            model, unsupervised_weight, ensembling_targets
        )

    return ensemble_precitions, loss_value, tape.gradient(loss_value, model.variables)

def pi_model_loss(X, y, U, pi_model, unsupervised_weight):
    z_labeled = pi_model(X)
    z_labeled_i = pi_model(X)

    z_unlabeled = pi_model(U)
    z_unlabeled_i = pi_model(U)

    # Loss = supervised loss + unsup loss of labeled sample + unsup loss unlabeled sample
    return tf.compat.v1.losses.softmax_cross_entropy(y, z_labeled) + \
        unsupervised_weight * (
            tf.keras.losses.mean_squared_error(z_labeled, z_labeled_i) +
            tf.keras.losses.mean_squared_error(z_unlabeled, z_unlabeled_i)
        )

def pi_model_gradients(X, y, U, pi_model, unsupervised_weight):

    with tf.GradientTape() as tape:
        loss_value = pi_model_loss(X, y, U, pi_model, unsupervised_weight)
    return loss_value, tape.gradient(loss_value, pi_model.variables)

# Ramps the value of the weight and learning rate in the first 80 epochs, according to the paper
def ramp_up_function(epoch, epoch_with_max_rampup=80):

    if epoch < epoch_with_max_rampup:
        p = max(0.0, float(epoch)) / float(epoch_with_max_rampup)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

# Ramps down the value of the learning rate and adam's beta in the last 50 epochs according to the paper
def ramp_down_function(epoch, num_epochs, epoch_with_max_rampdown = 50):

    if epoch >= (num_epochs - epoch_with_max_rampdown):
        p = (epoch - (num_epochs - epoch_with_max_rampdown)) * 0.5
        return math.exp(-(p * p) / epoch_with_max_rampdown)
    else:
        return 1.0

class _PiTeModel(tf.keras.Model):
    """
        Base class for pi and temporal ensembling models, as defined in
        https://research.nvidia.com/sites/default/files/publications/laine2017iclr_paper.pdf 

        the main difference between the two is what's used for the loss / gradient,
        although the prediction for the two does differ slightly
    """
    # TODO: add n_ins, n_outs
    def __init__(self):

        super(_PiTeModel, self).__init__() 

        # create the network with dropout as in the paper

        self._conv1a = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._conv1b = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._conv1c = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[3, 3],
                padding="same", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self._dropout1 = tf.keras.layers.Dropout(0.5)
        self._conv2a = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._conv2b = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._conv2c = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=[3, 3],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1),
            )
        )
        self._pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")
        self._dropout2 = tf.keras.layers.Dropout(0.5)
        self._conv3a = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=[3, 3],
                padding="valid", activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._conv3b = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=256, 
                kernel_size=[1, 1],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._conv3c = WeightNormalization(
            tf.keras.layers.Conv2D(
                filters=128, 
                kernel_size=[1, 1],
                padding="same", 
                activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )
        self._dense = WeightNormalization(
            tf.keras.layers.Dense(
                units=10, 
                activation=tf.nn.softmax,
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.keras.initializers.constant(0.1)
            )
        )

    def __gaussian_noise(self, input, std):

        noise = tf.random.normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32)
        return input + noise

    def __image_augmentation(self, image):

        random_shifts = np.random.randint(-2, 2, (image.numpy().shape[0], 2))
        random_transformations = tfa.image.translations_to_projective_transforms(random_shifts)
        image = tfa.image.transform(image, random_transformations, 'NEAREST', output_shape=tf.convert_to_tensor(image.numpy().shape[1:3], dtype=np.int32)) 

        return image

    def call(self, input, training=True):
        
        # add the stochastic augmentation to the input if we're training
        if training:
            h = self.__gaussian_noise(input, 0.15)
            h = self.__image_augmentation(h)
        else:
            h = input
        
        # pass the (augmented) input through the model
        h = self._conv1a(h, training)
        h = self._conv1b(h, training)
        h = self._conv1c(h, training)
        h = self._pool1(h)
        h = self._dropout1(h, training=training)

        h = self._conv2a(h, training)
        h = self._conv2b(h, training)
        h = self._conv2c(h, training)
        h = self._pool2(h)
        h = self._dropout2(h, training=training)

        h = self._conv3a(h, training)
        h = self._conv3b(h, training)
        h = self._conv3c(h, training)

        # Average Pooling
        h = tf.reduce_mean(h, axis=[1, 2])
        return self._dense(h, training)

class TemporalEnsembling():
    
    def __init__(self, 
            epochs, batch_size, max_lrate=0.0002,
            alpha=0.6, beta_1=[0.9,0.5], beta_2=0.98,
            max_unsupervised_weight=0.5
        ):

        self.checkpoint_directory = './checkpoints'

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_lrate = max_lrate

        self.lrate = tf.Variable(max_lrate)

        self.alpha = alpha

        self.beta_1 = tf.Variable(beta_1[0])
        self.beta_1_end = beta_1[1]
        self.beta_2 = beta_2

        self.max_unsupervised_weight = max_unsupervised_weight

        self.opt = tf.keras.optimizers.Adam(self.lrate, self.beta_1, self.beta_2)

        self.loss = temporal_ensembling_loss
        self.gradients = temporal_ensembling_gradients
        self.model = _PiTeModel()

    def fit(self, train_data, validation_data):
        # TODO: add n_ins, n_outs
        
        Z = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], 10))
        z = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], 10)) 

        sample_epoch = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], 1))

        # need to define a data generator...
        train_X_iterator = iter(range(train_data.X.shape[0]))
        train_U_iterator = iter(range(train_data.U.shape[0]))
        val_iterator = iter(range(validation_data.X.shape[0]))

        # custom training loop
        for epoch in range(self.epochs):
            rampdown_value = ramp_down_function(epoch, self.epochs)
            rampup_value = ramp_up_function(epoch)

            if epoch == 0:
                unsupervised_weight = 0
            else:
                unsupervised_weight = self.max_unsupervised_weight * rampup_value

            self.lrate.assign(rampup_value * rampdown_value * self.max_lrate)
            self.beta_1.assign(rampdown_value * self.beta_1[0] + (1.0 - rampdown_value) * self.beta_1[1])

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.Accuracy()
            epoch_loss_avg_val = tf.keras.metrics.Mean()
            epoch_accuracy_val = tf.keras.metrics.Accuracy()
            
            for _ in range(self.batch_size):

                X_indexes = next(train_X_iterator)
                U_indexes = next(train_U_iterator)
                X, y, U = train_data.X[X_indexes], train_data.y[X_indexes], train_data.U[U_indexes]

                ensemble_indexes = np.concatenate([X_indexes.numpy(), train_data.X.shape[0] + U_indexes.numpy()])
                ensemble_targets = z[ensemble_indexes]

                # this basically evals the model
                current_outputs, loss_val, grads = self.gradients(
                    X, y, U, self.model, unsupervised_weight, ensemble_targets
                )
                self.opt.apply_gradients(zip(grads, self.model.variables))

                # compute metrics...
                epoch_loss_avg(loss_val)
                epoch_accuracy(tf.argmax(self.model(X), 1), tf.argmax(y, 1))

                epoch_loss_avg(loss_val)
                epoch_accuracy(tf.argmax(self.model(X), 1), tf.argmax(y, 1))
                
                # update our ensemble stored outputs
                Z[ensemble_indexes, :] = self.alpha * Z[ensemble_indexes, :] + (1 - self.alpha) * current_outputs
                z[ensemble_indexes, :] = Z[ensemble_indexes, :] * (1. / (1. - self.alpha ** (sample_epoch[ensemble_indexes] + 1)))
                sample_epoch[ensemble_indexes] += 1

            # end of training batch - do a validation batch
            for _ in range(self.batch_size):
                X_val, y_val, _ = next(val_iterator)
                y_val_predictions = self.model(X_val, training=False)

                epoch_loss_avg_val(tf.compat.v1.losses.softmax_cross_entropy(y_val, y_val_predictions))
                epoch_accuracy_val(tf.argmax(y_val_predictions, 1), tf.argmax(y_val, 1))

            print("Epoch {:03d}/{:03d}: Train Loss: {:9.7f}, Train Accuracy: {:02.6%}, Validation Loss: {:9.7f}, "
              "Validation Accuracy: {:02.6%}, lr={:.9f}, unsupervised weight={:5.3f}, beta1={:.9f}".format(
                epoch+1,
                self.epochs,
                epoch_loss_avg.result(),
                epoch_accuracy.result(),
                epoch_loss_avg_val.result(),
                epoch_accuracy_val.result(),
                self.lrate.numpy(),
                unsupervised_weight,
                self.beta_1.numpy()
            ))

            # If the accuracy of validation improves save a checkpoint
            if best_val_accuracy < epoch_accuracy_val.result():
                best_val_accuracy = epoch_accuracy_val.result()
                checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
                checkpoint.save(file_prefix=self.checkpoint_directory)

        print('\nTrain Ended! Best Validation accuracy = {}\n'.format(best_val_accuracy))
        # Load the best model
        root = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
        root.restore(tf.train.latest_checkpoint(self.checkpoint_directory))

    def predict(self, X):
        
        return self.model(X, training=False)
