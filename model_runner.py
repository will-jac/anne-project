# from scratch tensorflow implementation of temporal ensembling 
#   since I couldn't even get theano / lasagne to import, in part due to the discontinuation
#   of development on theano, I thought this would be a more principled approach.
import os
import math
import numpy as np 
import tensorflow as tf

import tensorflow_addons as tfa

from generator import training_generator

def gaussian_noise(inputs, std):

    noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=std, dtype=tf.float32)
    return inputs + noise

# for more information about how to use the loss / gradients defined here, see
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

def temporal_ensembling_loss(X, y, U, model, unsupervised_weight, ensembling_targets):
    z_X = model(X)
    z_U = model(U)

    pred = tf.concat([z_X, z_U], 0)

    return pred, tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y, z_X)) + \
        unsupervised_weight * \
        tf.reduce_sum(tf.keras.losses.mean_squared_error(ensembling_targets, pred))

def temporal_ensembling_gradients(X, y, U, model, unsupervised_weight, ensembling_targets):

    with tf.GradientTape() as tape:
        ensemble_precitions, loss_value = temporal_ensembling_loss( 
            X, y, U, model, unsupervised_weight, ensembling_targets
        )

    return ensemble_precitions, loss_value, tape.gradient(loss_value, model.trainable_weights)

def pi_loss(X, y, U, model, unsupervised_weight):
    z_labeled = model(X)
    # additive gaussian noise added to one branch
    z_labeled_i = model(gaussian_noise(X, 0.1))

    z_unlabeled = model(U)
    z_unlabeled_i = model(gaussian_noise(U, 0.1))
    # Loss = supervised loss + unsup loss of labeled sample + unsup loss unlabeled sample
    # print(X.shape, z_labeled.shape, U.shape)
    # print('y',y)
    # print('pred',z_labeled)
    # print(tf.keras.losses.categorical_crossentropy(y, z_labeled))
    # print(tf.keras.losses.mean_squared_error(z_labeled, z_labeled_i))
    # print(tf.keras.losses.mean_squared_error(z_unlabeled, z_unlabeled_i))
    # print(z_labeled.shape, z_labeled_i.shape, y.shape, z_unlabeled.shape, z_unlabeled_i.shape)
    return tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y, z_labeled)) + unsupervised_weight * (
        tf.reduce_mean(tf.keras.losses.mean_squared_error(z_labeled, z_labeled_i)) +
        tf.reduce_mean(tf.keras.losses.mean_squared_error(z_unlabeled, z_unlabeled_i))
    )

def pi_gradients(X, y, U, model, unsupervised_weight, ensembling_targets=None):

    with tf.GradientTape() as tape:
        loss = pi_loss(X, y, U, model, unsupervised_weight)
        # print('pi model loss:', pi_loss)
    
    # None to preserve compatibility with temporal ensembling
    return None, loss, tape.gradient(loss, model.trainable_weights)

def pl_loss(X, y, U, model, unsupervised_weight):
    y_pred_labeled = model(X)
    y_pred_unlabeled = model(X)

    y_pl = tf.one_hot(tf.argmax(y_pred_unlabeled, axis=-1), y.shape[1])
    y_pl = tf.cast(y_pl, y.dtype)

    return tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y, y_pred_labeled)) + \
        unsupervised_weight * tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_pl, y_pred_unlabeled))

def pl_gradients(X, y, U, model, unsupervised_weight, ensembling_targets=None):
    with tf.GradientTape() as tape:
        loss = pl_loss(X, y, U, model, unsupervised_weight)
        # print('pi model loss:', pi_loss)
    
    # None to preserve compatibility with temporal ensembling
    return None, loss, tape.gradient(loss, model.trainable_weights)

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

# unsupervised weight scheduler for pseudo labels
def alpha_schedule(epoch, T1=100, T2=600, af=3.0):
    if epoch < T1:
        # print('alpha set to 0.0')
        return 0.0
    elif epoch < T2:
        # print('alpha set to', (epoch_num - self.T1) / (self.T2 - self.T1) * self.af)
        return (epoch - T1) / (T2 - T1) * af
    else:
        # print('alpha set to', self.af)
        return af

class _ModelBase():
    
    def __init__(self, model, args=None,
            epochs=1000, 
            minibatch_size=100, 
            min_labeled_per_minibatch=None,
            minibatches_per_epoch=1,
            steps_per_minibatch=2,
            patience=500,
            lrate=0.0002,
            ramp_up=50,
            ramp_down=80,
            alpha=0.6, beta_1=[0.9,0.5], beta_2=0.98,
            max_unsupervised_weight=0.5,
            use_image_augmentation=False,
            num_classes=10
    ):

        self.checkpoint_directory = './checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_directory, "ckpt")

        # pre-fill with defaults
        self.epochs = self.max_epochs = epochs
        self.minibatch_size = minibatch_size
        self.min_labeled_per_minibatch = min_labeled_per_minibatch
        self.minibatches_per_epoch = minibatches_per_epoch
        self.steps_per_minibatch = steps_per_minibatch
        self.patience=patience
        self.epochs_without_improvement = 0

        self.ramp_up=ramp_up
        self.ramp_down=ramp_down

        self.max_lrate = lrate

        self.alpha = alpha

        self.max_unsupervised_weight = max_unsupervised_weight

        self.num_classes=num_classes
        
        # add the args
        if 'epochs' in args:
            self.epochs = self.max_epochs = args['epochs']
        if 'minibatch_size' in args:
            self.minibatch_size = args['minibatch_size']
        if 'min_labeled_per_minibatch' in args:
            self.min_labeled_per_minibatch = args['min_labeled_per_minibatch']
        if 'minibatches_per_epoch' in args:
            self.minibatches_per_epoch = args['minibatches_per_epoch']
        if 'steps_per_minibatch' in args:
            self.steps_per_minibatch = args['steps_per_minibatch']
        if 'patience' in args:
            self.patience=args['patience']
        if 'ramp_up' in args:
            self.ramp_up=args['ramp_up']
        if 'ramp_down' in args:
            self.ramp_down=args['ramp_down']
        if 'lrate' in args:
            self.max_lrate = args['lrate']
        if 'alpha' in args:
            self.alpha = args['alpha']
        if 'max_unsupervised_weight' in args:
            self.max_unsupervised_weight = args['max_unsupervised_weight']
        if 'num_classes' in args:
            self.num_classes = args['num_classes']

        self.lrate = tf.Variable(lrate, trainable=False)

        self.beta_1 = tf.Variable(beta_1[0], trainable=False)
        self.beta_1_start = beta_1[0]
        self.beta_1_end = beta_1[1]
        self.beta_2 = beta_2

        self.opt = tf.keras.optimizers.Adam(self.lrate, self.beta_1, self.beta_2)

        self.model = model(do_image_augmentation = use_image_augmentation)

        self.te = False
        self.pi = False
        self.pl = False

    def fit(self, train_data, validation_data):
        
        if self.te:
            Z = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], self.num_classes))
            z = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], self.num_classes)) 

            sample_epoch = np.zeros((train_data.X.shape[0] + train_data.U.shape[0], 1))

        if self.pi:
            if self.min_labeled_per_minibatch is None:
                unsupervised_weight = float(80 * train_data.X.shape[0]) / (train_data.X.shape[0] + train_data.U.shape[0])
            else:
                unsupervised_weight = float(80 * self.min_labeled_per_minibatch) / self.minibatch_size

        # define a data generator
        # self.num_batches = (train_data.X.shape[0]  + train_data.U.shape[0]) // self.batch_size 
        generator = training_generator(train_data, self.minibatch_size, self.min_labeled_per_minibatch)
        
        best_val_accuracy = 0.0
        epochs_without_improvement = 0

        train_acc = []
        val_acc = []

        # custom training loop
        for epoch in range(self.max_epochs):
            rampdown_value = ramp_down_function(epoch, self.epochs, self.ramp_down)
            rampup_value = ramp_up_function(epoch, self.ramp_up)

            if self.te:
                if epoch == 0:
                    unsupervised_weight = 0
                else:
                    unsupervised_weight = self.max_unsupervised_weight * rampup_value
            if self.pl:
                unsupervised_weight = alpha_schedule(epoch)

            self.lrate.assign(rampup_value * rampdown_value * self.max_lrate)
            self.beta_1.assign(rampdown_value * self.beta_1_start + (1.0 - rampdown_value) * self.beta_1_end)

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
            epoch_loss_avg_validation = 0.0
            epoch_accuracy_validation = tf.keras.metrics.CategoricalAccuracy()

            for _ in range(self.minibatches_per_epoch):
                # optimize to this training data
                Xi, Ui = next(generator)
                X, y, U = train_data.X[Xi], train_data.y[Xi], train_data.U[Ui]
                # perform a minibatch
                for _ in range(self.steps_per_minibatch):

                    if self.te:
                        ensemble_indexes = np.concatenate([Xi, train_data.X.shape[0] + Ui])
                        ensemble_targets = z[ensemble_indexes]
                    else:
                        ensemble_indexes = None
                        ensemble_targets = None

                    # this evals the model
                    current_outputs, loss_val, grads = self.gradients(
                        X, y, U, self.model, unsupervised_weight, ensemble_targets
                    )
                    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

                    # compute metrics...
                    epoch_loss_avg(loss_val)
                    epoch_accuracy(self.model(X, training=False), y)
                    
                    # update our ensemble stored outputs
                    if self.te:
                        Z[ensemble_indexes, :] = self.alpha * Z[ensemble_indexes, :] + (1 - self.alpha) * current_outputs
                        z[ensemble_indexes, :] = Z[ensemble_indexes, :] * (1. / (1. - self.alpha ** (sample_epoch[ensemble_indexes] + 1)))
                        sample_epoch[ensemble_indexes] += 1

            # end of training batch - do a validation batch
            y_validation_predictions = self.model(validation_data.X, training=False)

            loss_validation_val = tf.keras.losses.categorical_crossentropy(validation_data.y, y_validation_predictions)
            epoch_loss_avg_validation = tf.reduce_mean(loss_validation_val)

            epoch_accuracy_validation(y_validation_predictions, validation_data.y)

            train_acc.append(float(epoch_accuracy.result()))
            val_acc.append(float(epoch_accuracy_validation.result()))

            print("Epoch {:03d}/{:03d}: Train Loss: {:5.2f}, Train Accuracy: {:02.2%}, Validation Loss: {:5.2f}, "
              "Validation Accuracy: {:02.2%}, lr={:.9f}, unsupervised weight={:5.3f}, beta1={:.9f}".format(
                epoch,
                self.max_epochs,
                epoch_loss_avg.result(),
                epoch_accuracy.result(),
                epoch_loss_avg_validation,
                epoch_accuracy_validation.result(),
                self.lrate.numpy(),
                unsupervised_weight,
                self.beta_1.numpy()
            ))

            self.epochs_without_improvement += 1
            # If the accuracy of validation improves save a checkpoint
            if best_val_accuracy < epoch_accuracy_validation.result():
                self.epochs_without_improvement = 0
                best_val_accuracy = epoch_accuracy_validation.result()
                checkpoint = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
                checkpoint.save(file_prefix=self.checkpoint_prefix)

            if self.epochs_without_improvement > self.patience:
                # no improvement. Instead of quitting, start ramping down
                print('starting ramp down to ',epoch + self.ramp_down)
                self.epochs = epoch + self.ramp_down
                self.patience = np.inf # avoid doing this again

            if epoch >= self.epochs:
                print('stopping due to ramp down')
                # stop!
                break

        print('\nTrain Ended! Best Validation accuracy = {}\n'.format(best_val_accuracy))
        # Load the best model
        root = tf.train.Checkpoint(optimizer=self.opt, model=self.model)
        root.restore(tf.train.latest_checkpoint(self.checkpoint_prefix))

        return {'train':train_acc, 'val':val_acc}

    def predict(self, X):
        return self.model(X, training=False)

class TemporalEnsembling(_ModelBase):
    def __init__(self, model, args):
        super().__init__(model, args)

        self.name = 'temporal ensembling'
        self.te = True
        self.loss = temporal_ensembling_loss
        self.gradients = temporal_ensembling_gradients 

class PiModel(_ModelBase):
    
    def __init__(self, model, args):
        super().__init__(model, args)
        self.name = 'pi'
        self.pi = True
        self.loss = pi_loss
        self.gradients = pi_gradients 


class PseudoLabels(_ModelBase):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.name = 'Pseudo Labels'
        self.pl = True
        self.loss = pl_loss
        self.gradients = pl_gradients 
