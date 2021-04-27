# from scratch tensorflow implementation of temporal ensembling 
#   since I couldn't even get theano / lasagne to import, in part due to the discontinuation
#   of development on theano, I thought this would be a more principled approach.

import math
import numpy as np 
import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras import backend as K

from generator import training_generator

# for more information about how to use the loss / gradients defined here, see
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

def temporal_ensembling_loss(X, y, U, model, unsupervised_weight, ensembling_targets):
    z_X = model(X)
    z_U = model(U)

    pred = tf.concat([z_X, z_U], 0)

    return pred, K.sum(tf.keras.losses.categorical_crossentropy(y, z_X)) + \
            unsupervised_weight * K.sum(tf.keras.losses.mean_squared_error(ensembling_targets, pred))

def temporal_ensembling_gradients(X, y, U, model, unsupervised_weight, ensembling_targets):

    with tf.GradientTape() as tape:
        ensemble_precitions, loss_value = temporal_ensembling_loss( 
            X, y, U,
            model, unsupervised_weight, ensembling_targets
        )

    return ensemble_precitions, loss_value, tape.gradient(loss_value, model.trainable_weights)

def pi_model_loss(X, y, U, pi_model, unsupervised_weight):
    z_labeled = pi_model(X)
    z_labeled_i = pi_model(X)

    z_unlabeled = pi_model(U)
    z_unlabeled_i = pi_model(U)

    # Loss = supervised loss + unsup loss of labeled sample + unsup loss unlabeled sample

    # print(z_labeled.shape, z_labeled_i.shape, y.shape, z_unlabeled.shape, z_unlabeled_i.shape)
    return K.sum(tf.keras.losses.categorical_crossentropy(y, z_labeled)) + \
        unsupervised_weight * (
            K.sum(tf.keras.losses.mean_squared_error(z_labeled, z_labeled_i)) + \
            K.sum(tf.keras.losses.mean_squared_error(z_unlabeled, z_unlabeled_i))
        )

def pi_model_gradients(X, y, U, pi_model, unsupervised_weight):

    with tf.GradientTape() as tape:
        loss_value = pi_model_loss(X, y, U, pi_model, unsupervised_weight)
    return loss_value, tape.gradient(loss_value, pi_model.trainable_weights)

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


class TemporalEnsembling():
    
    def __init__(self, model,
            epochs, batch_size, max_lrate=0.0002,
            alpha=0.6, beta_1=[0.9,0.5], beta_2=0.98,
            max_unsupervised_weight=0.5
        ):

        self.checkpoint_directory = './checkpoints'

        self.epochs = epochs
        self.batch_size = batch_size
        
        self.max_lrate = max_lrate

        self.lrate = tf.Variable(max_lrate, trainable=False)

        self.alpha = alpha

        self.beta_1 = tf.Variable(beta_1[0], trainable=False)
        self.beta_1_start = beta_1[0]
        self.beta_1_end = beta_1[1]
        self.beta_2 = beta_2

        self.max_unsupervised_weight = max_unsupervised_weight

        self.opt = tf.keras.optimizers.Adam(self.lrate, self.beta_1, self.beta_2)

        self.loss = temporal_ensembling_loss
        self.gradients = temporal_ensembling_gradients
        self.model = model

    def fit(self, train_data, validation_data):
        # TODO: add n_ins, n_outs
        
        Z = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], 10))
        z = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], 10)) 

        sample_epoch = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], 1))

        # define a data generator
        self.num_batches = (train_data.X.shape[0]  + train_data.U.shape[0]) // self.batch_size 
        generator = training_generator(train_data.X, train_data.y, train_data.U, train_data.X.shape[0] / train_data.U.shape[0], self.batch_size)

        # custom training loop
        for epoch in range(self.epochs):
            rampdown_value = ramp_down_function(epoch, self.epochs)
            rampup_value = ramp_up_function(epoch)

            if epoch == 0:
                unsupervised_weight = 0
            else:
                unsupervised_weight = self.max_unsupervised_weight * rampup_value

            self.lrate.assign(rampup_value * rampdown_value * self.max_lrate)
            self.beta_1.assign(rampdown_value * self.beta_1_start + (1.0 - rampdown_value) * self.beta_1_end)

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.Accuracy()
            epoch_loss_avg_validation = tf.keras.metrics.Mean()
            epoch_accuracy_validation = tf.keras.metrics.Accuracy()
            
            Xi, Ui = next(generator)
            X, y, U = train_data.X[Xi], train_data.y[Xi], train_data.U[Ui]

            for _ in range(self.num_batches):

                ensemble_indexes = np.concatenate([Xi, train_data.X.shape[0] + Ui])
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
            y_validation_predictions = self.model(validation_data.X, training=False)

            epoch_loss_avg_validation(tf.compat.v1.losses.softmax_cross_entropy(validation_data.y, y_validation_predictions))
            epoch_accuracy_validation(tf.argmax(y_validation_predictions, 1), tf.argmax(validation_data.y, 1))

            print("Epoch {:03d}/{:03d}: Train Loss: {:9.7f}, Train Accuracy: {:02.6%}, Validation Loss: {:9.7f}, "
              "Validation Accuracy: {:02.6%}, lr={:.9f}, unsupervised weight={:5.3f}, beta1={:.9f}".format(
                epoch+1,
                self.epochs,
                epoch_loss_avg.result(),
                epoch_accuracy.result(),
                epoch_loss_avg_validation.result(),
                epoch_accuracy_validation.result(),
                self.lrate.numpy(),
                unsupervised_weight,
                self.beta_1.numpy()
            ))

            # If the accuracy of validation improves save a checkpoint
            if best_val_accuracy < epoch_accuracy_validation.result():
                best_val_accuracy = epoch_accuracy_validation.result()
                checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
                checkpoint.save(file_prefix=self.checkpoint_directory)

        print('\nTrain Ended! Best Validation accuracy = {}\n'.format(best_val_accuracy))
        # Load the best model
        root = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
        root.restore(tf.train.latest_checkpoint(self.checkpoint_directory))

    def predict(self, X):
        return self.model(X, training=False)

class PiModel():
    
    def __init__(self, model,
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
        self.beta_1_start = beta_1[0]
        self.beta_1_end = beta_1[1]
        self.beta_2 = beta_2

        self.max_unsupervised_weight = max_unsupervised_weight

        self.opt = tf.keras.optimizers.Adam(self.lrate, self.beta_1, self.beta_2)

        self.loss = pi_model_loss
        self.gradients = pi_model_gradients
        self.model = model()

    def fit(self, train_data, validation_data):
        # TODO: add n_ins, n_outs

        # define a data generator
        self.num_batches = int((train_data.X.shape[0]  + train_data.U.shape[0]) / self.batch_size)
        generator = training_generator(train_data.X, train_data.y, train_data.U, train_data.X.shape[0] / train_data.U.shape[0], self.batch_size)

        # custom training loop

        best_val_accuracy = 0.0

        print('training!')

        for epoch in range(self.epochs):
            print('epoch:', epoch)
            rampdown_value = ramp_down_function(epoch, self.epochs)
            rampup_value = ramp_up_function(epoch)

            if epoch == 0:
                unsupervised_weight = 0
            else:
                unsupervised_weight = self.max_unsupervised_weight * rampup_value

            self.lrate.assign(rampup_value * rampdown_value * self.max_lrate)
            self.beta_1.assign(rampdown_value * self.beta_1_start + (1.0 - rampdown_value) * self.beta_1_end)

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.Accuracy()
            epoch_loss_avg_validation = tf.keras.metrics.Mean()
            epoch_accuracy_validation = tf.keras.metrics.Accuracy()

            for _ in range(self.num_batches):

                Xi, Ui = next(generator)
                X, y, U = train_data.X[Xi], train_data.y[Xi], train_data.U[Ui]

                # this basically evals the model
                loss_val, grads = self.gradients(
                    X, y, U, self.model, unsupervised_weight
                )
                self.opt.apply_gradients(zip(grads, self.model.trainable_weights))

                # compute metrics...
                epoch_loss_avg(loss_val)
                epoch_accuracy(tf.argmax(self.model(X), 1), tf.argmax(y, 1))

                epoch_loss_avg(loss_val)
                epoch_accuracy(tf.argmax(self.model(X), 1), tf.argmax(y, 1))

            # end of training batch - do a validation batch
            y_validation_predictions = self.model(validation_data.X, training=False)

            epoch_loss_avg_validation(tf.compat.v1.losses.softmax_cross_entropy(validation_data.y, y_validation_predictions))
            epoch_accuracy_validation(tf.argmax(y_validation_predictions, 1), tf.argmax(validation_data.y, 1))

            print("Epoch {:03d}/{:03d}: Train Loss: {:9.7f}, Train Accuracy: {:02.6%}, Validation Loss: {:9.7f}, "
              "Validation Accuracy: {:02.6%}, lr={:.9f}, unsupervised weight={:5.3f}, beta1={:.9f}".format(
                epoch+1,
                self.epochs,
                epoch_loss_avg.result(),
                epoch_accuracy.result(),
                epoch_loss_avg_validation.result(),
                epoch_accuracy_validation.result(),
                self.lrate.numpy(),
                unsupervised_weight,
                self.beta_1.numpy()
            ))

            # If the accuracy of validation improves save a checkpoint
            if best_val_accuracy < epoch_accuracy_validation.result():
                best_val_accuracy = epoch_accuracy_validation.result()
                checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
                checkpoint.save(file_prefix=self.checkpoint_directory)

        print('\nTrain Ended! Best Validation accuracy = {}\n'.format(best_val_accuracy))
        # Load the best model
        root = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
        root.restore(tf.train.latest_checkpoint(self.checkpoint_directory))

    def predict(self, X):
        return self.model(X, training=False)
