
import pickle
import numpy as np
import os

def load_cifar_10(one_hot = True):

    # Function to load each file
    def load_cifar_batches(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            with open(os.path.join('data', 'cifar-10', fn), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            images.append(np.asarray(data[b'data'], dtype='float32').reshape(-1, 3, 32, 32) / np.float32(255))
            labels.append(np.asarray(data[b'labels'], dtype='int32'))
        return np.concatenate(images), np.concatenate(labels)

    X_train, y_train = load_cifar_batches(['data_batch_%d' % i for i in (1, 2, 3, 4, 5)])
    X_test, y_test = load_cifar_batches('test_batch')

    if one_hot:
        y_train = np.eye(10)[y_train.reshape(-1)]
        y_test = np.eye(10)[y_test.reshape(-1)]
    # We have our train test split, now we just need to seperate X_train, y_train into X_train, y_train, U_train...

    return X_train, y_train, X_test, y_test

