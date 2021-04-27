import random

def training_generator(X, y, U, labeled_unlabeled_ratio, batch_size=10):
    '''
    Generator for producing random mini-batches of training samples, staying proportional to the number of labeled / unlabeled samples
    
    :param ins: Full set of training set inputs (examples x row x col x chan)
    :param outs: Corresponding set of sample (examples x nclasses)
    :param batch_size: Number of samples for each minibatch
    :param input_name: Name of the model layer that is used for the input of the model
    :param output_name: Name of the model layer that is used for the output of the model
    '''
    
    while True:
        # Randomly select a set of example indices
        Xi = random.choices(range(X.shape[0]), k=int(batch_size*labeled_unlabeled_ratio))
        Ui = random.choices(range(U.shape[0]), k=int(batch_size*(1/labeled_unlabeled_ratio)))

        # The generator will produce a pair of return values: one for inputs and one for outputs
        yield(Xi, Ui)
        