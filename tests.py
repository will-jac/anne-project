import numpy as np
import util

# data sets
import data.cifar_10 as cifar_10
import tensorflow_datasets as tfds

# other things
import tensorflow as tf
from sklearn.metrics import roc_curve, plot_roc_curve

# All test functions need to 
# Import the data and set-up train, test, validation sets.
# Call model.fit on the training and validation test.
# Return the accuracy based on the test set. 
# test-specific models

from base_models import *

def whiten_norm(x):
    x = x - np.mean(x, axis=(1, 2, 3), keepdims=True)
    x = x / (np.mean(x ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)
    return x

def adult_test(model, num_label=1300):

    with tf.device('/CPU:0'):
        import data.adult as adult
        # turn the labels into a proper matrix
        adult.y = np.reshape(adult.y, (adult.y.shape[0], 1))   
        # one-hot encode the outs
        adult.y = np.eye(2)[adult.y.reshape(-1)]
        print(adult.X.shape, adult.y.shape)

        # (label, unlabeled) = util.train_test_valid_split(adult.X, adult.y, split=(u, 1 - u))

        # (train, test, valid) = util.train_test_valid_split(label.X, label.y, split=(0.6, 0.2, 0.2), U = unlabeled.X)

        # Split training test into labeled and unlabeled
        train = util.label_unlabel_split(adult.X, adult.y, num_label, 2)    
        print(train.X.shape, train.y.shape, train.U.shape)

        # Split training data into training and validation
        (train, valid, test) = util.train_test_valid_split(train.X, train.y, num_classes=2, split=(0.8, 0.1, 0.1), U = train.U)

        print('TR:', train.X.shape, train.y.shape, train.U.shape)
        print('v', valid.X.shape, valid.y.shape)

    with tf.device('/GPU:0'):
        # for model, name in zip(models, model_names):
        hist = model.fit(train, valid)

        y_pred = model.predict(test.X)
        # y_pred = y_pred.ravel() 
        
        acc = float(tf.reduce_mean(tf.keras.metrics.categorical_accuracy(test.y[0:1000], y_pred)))
        print(model.name, ' : acc:', acc)

    # classify
    # for i, y_p in enumerate(y_pred):
    #     if y_p > 0.0: # TODO: should be 0?
    #         y_pred[i] = 1
    #     else:
    #         y_pred[i] = -1
    # wrong = util.percent_wrong(y_pred.ravel(), test.y.ravel())
    # acc = 1.0 - wrong
    # print(model.name, ' : acc:', acc)

    return model, {'hist':hist, 'acc':acc}

def cifar10_test (model, num_label=4000):

    # load data on the cpu
    with tf.device('/CPU:0'):

        # Load in training and test data
        X_train, y_train = tfds.as_numpy(tfds.load('cifar10', split='train', as_supervised=True, batch_size=-1)) #cifar_10.load_cifar_10()
        X_test, y_test = tfds.as_numpy(tfds.load('cifar10', split='test', as_supervised=True, batch_size=-1))
        
        # one-hot encode the outs
        y_train = np.eye(10)[y_train.reshape(-1)]
        # print('y_train sample:', y_train[0:10])
        y_test = np.eye(10)[y_test.reshape(-1)]
        # print('y_test sample:', y_test[0:10])
        
        # cast it all to floats for image augmentation, rescale to [0,1]
        X_train = X_train.astype('float32') / np.float(255.0)
        X_test = X_test.astype('float32') / np.float(255.0)

        # whiten the data or apply zca
        X_train = whiten_norm(X_train)
        X_test  = whiten_norm(X_test)
        # X_train = whiten_norm(X_train)
        # X_test  = whiten_norm(X_test)

        # X_train, y_train, X_test, y_test = cifar_10.load_cifar_10()

        print('loaded cifar10', X_train.shape, X_test.shape)
        # Setup test set
        test = util.Data(X_test, y_test, None)
        
        # Split training test into labeled and unlabeled
        train = util.label_unlabel_split(X_train, y_train, num_label, 10)

        # Split training data into training and validation
        (train, valid) = util.train_test_valid_split(train.X, train.y, split=(0.9, 0.1), U = train.U)

        # One-hot encode cifar_10.y_train and cifar_10.y_test?
        ## ^^ yes. Done.
        print('TR:', train.X.shape, train.y.shape, train.U.shape)
        print('v', valid.X.shape, valid.y.shape)

    # fit on the gpu
    with tf.device('/GPU:0'):

        # Train model using training and validation sets
        hist = model.fit(train, valid)
    
    print('evaluating on (subset) of test set...')
    with tf.device('/CPU:0'):
        # Test the model using test set
        y_pred = model.predict(test.X[0:1000])

        # if outputs are one-hot encoded, need to decode for correctness test
        # wrong = util.percent_wrong(y_pred, test.y)
        # acc = 1.0 - wrong
        acc = float(tf.reduce_mean(tf.keras.metrics.categorical_accuracy(test.y[0:1000], y_pred)))
        print(model.name, ' : acc:', acc)

    return model, {'hist':hist, 'acc':acc}

def svhn_test (model, num_label=500):
    # Load SVHN dataset
    # load data on the cpu
    with tf.device('/CPU:0'):
        X_train, y_train = tfds.as_numpy(tfds.load('svhn_cropped', split='train', as_supervised=True, batch_size=-1))
        X_test, y_test = tfds.as_numpy(tfds.load('svhn_cropped', split='test', as_supervised=True, batch_size=-1))

        # one-hot encode the outs
        y_train = np.eye(10)[y_train.reshape(-1)]
        y_test = np.eye(10)[y_test.reshape(-1)]

        # cast it all to floats for image augmentation, rescale to [0,1]
        X_train = X_train.astype('float32') / np.float(255.0)
        X_test = X_test.astype('float32') / np.float(255.0)
        
        # whiten the data or apply zca
        X_train = whiten_norm(X_train)
        X_test  = whiten_norm(X_test)

        print('loaded svfn', X_train.shape, X_test.shape)
        # Setup test set
        test = util.Data(X_test, y_test, None)
        
        # Split training test into labeled and unlabeled
        train = util.label_unlabel_split(X_train, y_train, num_label, 10)

        # Split training data into training and validation
        (train, valid) = util.train_test_valid_split(train.X, train.y, split=(0.9, 0.1), U = train.U)

        print('TR:', train.X.shape, train.y.shape, train.U.shape)
        print('v', valid.X.shape, valid.y.shape)
    # fit on the gpu
    with tf.device('/GPU:0'):
        # Train model using training and validation sets
        hist = model.fit(train, valid)

    print('evaluating on (subset) of test set...')
    with tf.device('/CPU:0'):
        # Test the model using test set
        y_pred = model.predict(test.X[0:1000])

        acc = float(tf.reduce_mean(tf.keras.metrics.categorical_accuracy(test.y[0:1000], y_pred)))
        print(model.name, ' : acc:', acc)

    return model, {'hist':hist, 'acc':acc}
    
def mnist_test (model, num_label=100):
    # Load SVHN dataset
    # load data on the cpu
    with tf.device('/CPU:0'):
        X_train, y_train = tfds.as_numpy(tfds.load('mnist', split='train', as_supervised=True, batch_size=-1))
        X_test, y_test = tfds.as_numpy(tfds.load('mnist', split='test', as_supervised=True, batch_size=-1))

        # one-hot encode the outs
        y_train = np.eye(10)[y_train.reshape(-1)]
        y_test = np.eye(10)[y_test.reshape(-1)]

        # cast it all to floats for image augmentation, rescale to [0,1]
        X_train = X_train.astype('float32') / np.float(255.0)
        X_test = X_test.astype('float32') / np.float(255.0)
        
        # whiten the data or apply zca
        X_train = whiten_norm(X_train)
        X_test  = whiten_norm(X_test)

        print('loaded mnist', X_train.shape, X_test.shape)
        # Setup test set
        test = util.Data(X_test, y_test, None)
        
        # Split training test into labeled and unlabeled
        train = util.label_unlabel_split(X_train, y_train, num_label, 10)

        # Split training data into training and validation
        (train, valid) = util.train_test_valid_split(train.X, train.y, split=(0.9, 0.1), U = train.U)

        print('TR:', train.X.shape, train.y.shape, train.U.shape)
        print('v', valid.X.shape, valid.y.shape)

    # fit on the gpu
    with tf.device('/GPU:0'):
        # Train model using training and validation sets
        hist = model.fit(train, valid)

    print('evaluating on (subset) of test set...')
    with tf.device('/CPU:0'):
        # Test the model using test set
        y_pred = model.predict(test.X[0:1000])

        acc = float(tf.reduce_mean(tf.keras.metrics.categorical_accuracy(test.y[0:1000], y_pred)))
        print(model.name, ' : acc:', acc)
    return model, {'hist':hist, 'acc':acc}

# return test-specific model and the testing function (which should accept a model and return the accuracy)
tests = {
    'cifar10' : (Cifar10Model, cifar10_test),
    'cifar10_experimental' : (cifar10_model, cifar10_test),
    'cifar10_experimental_pretrain' : (cifar10_model_pretrain, cifar10_test),
    # can keep using the cifar10 model because we have 10 outputs and 32 x 32 inputs for all
    'svhn' : (cifar10_model, svhn_test),
    'mnist' : (mnist_model, mnist_test),
    'adult' : (adult_model, adult_test)
}