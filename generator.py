import random
import numpy as np

def training_generator(X, y, U, labeled_to_total_ratio, batch_size=10):
    '''
    Generator for producing deterministic mini-batches of training samples, staying proportional to the number of labeled / unlabeled samples
    
    :param ins: Full set of training set inputs (examples x row x col x chan)
    :param outs: Corresponding set of sample (examples x nclasses)
    :param batch_size: Number of samples for each minibatch
    :param input_name: Name of the model layer that is used for the input of the model
    :param output_name: Name of the model layer that is used for the output of the model
    '''
    x_index = u_index = 0
    x_size = int(round(batch_size*(labeled_to_total_ratio)))
    u_size = int(round(batch_size*(1-labeled_to_total_ratio)))
    x_end = X.shape[0] - 1
    u_end = U.shape[0] - 1

    print('creating generator..', 'batch', batch_size, 'ratio', labeled_to_total_ratio, 'sizes', x_size, u_size, 'ends', x_end, u_end)
    while True:
        # Randomly select a set of example indices
        Xi = np.arange(x_index, min(x_index + x_size, X.shape[0]))
        Ui = np.arange(u_index, min(u_index + u_size, U.shape[0]))

        if x_index + x_size - x_end > 0:
            x_index += x_size
        else:
            x_index = 0
            
        if u_index + u_size - u_end > 0:
            u_index += u_size
        else:
            u_index = 0

        # The generator will produce a pair of return values: one for inputs and one for outputs
        yield(Xi, Ui)
        