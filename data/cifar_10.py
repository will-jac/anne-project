
import pickle
import numpy as np
import os

# Call function to load data
X_train, y_train, X_text, y_test = load_cifar_10()

def load_cifar_10():

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

    return X_train, y_train, X_test, y_test
