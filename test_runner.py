import numpy as np
import pickle as pkl
import argparse

# ML methods 
import models
import tests

from types import SimpleNamespace

configs = {
    'adult_pi' :{
        'method' : 'pi',
        'test' : 'adult',
        'dir' : './results/adult/pi',
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'num_classes' : 2
    },
    'adult_te' : {
        'method' : 'te',
        'test' : 'adult',
        'dir' : './results/adult/te',
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'num_classes' : 2
    },
    'adult_pl' : {
        'method' : 'pl',
        'test' : 'adult',
        'dir' : './results/adult/pl',
        'lrate' : 0.0001,
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,   
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'use_dae' : False,
        'num_classes' : 2
    },


    'mnist_pi' :{
        'method' : 'pi',
        'test' : 'mnist',
        'dir' : './results/mnist/pi',
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'num_classes' : 10
    },
    'mnist_te' : {
        'method' : 'te',
        'test' : 'mnist',
        'dir' : './results/mnist/te',
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'num_classes' : 10
    },
    'mnist_pl' : {
        'method' : 'pl',
        'test' : 'mnist',
        'dir' : './results/mnist/pl',
        'lrate' : 0.0001,
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,   
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'use_dae' : False,
        'num_classes' : 10
    },


    'svhn_pi' :{
        'method' : 'pi',
        'test' : 'svhn',
        'dir' : './results/svhn/pi',
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'num_classes' : 10
    },
    'svhn_te' : {
        'method' : 'te',
        'test' : 'svhn',
        'dir' : './results/svhn/te',
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'num_classes' : 10
    },
    'svhn_pl' : {
        'method' : 'pl',
        'test' : 'svhn',
        'dir' : './results/svhn/pl',
        'lrate' : 0.0001,
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : None,
        'patience' : 100,   
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'use_dae' : False,
        'num_classes' : 10
    },

    'cifar10pi' :{
        'method' : 'pi',
        'test' : 'cifar10_experimental',
        'dir' : './results/cifar10/pi',
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : 100,
        'patience' : 100,
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'num_classes' : 10
    },
    'cifar10te' : {
        'method' : 'te',
        'test' : 'cifar10_experimental',
        'dir' : './results/cifar10/te',
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : 100,
        'patience' : 100,
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'num_classes' : 10
    },
    'cifar10pl' : {
        'method' : 'pl',
        'test' : 'cifar10_experimental',
        'dir' : './results/cifar10/pl',
        'lrate' : 0.0001,
        'epochs' : 500,
        'minibatch_size' : 200,
        'minibatches_per_epoch' : 100,
        'min_labeled_per_minibatch' : 100,
        'patience' : 100,   
        'steps_per_minibatch' : 1,
        'use_image_augmentation' : True,
        'use_dae' : False,
        'num_classes' : 10
    },
    # not working, and honestly, I don't want to put the work in to make it work
    'cifar10pl_pretrain' : {
        'method' : 'pl',
        'test' : 'cifar10_experimental_pretrain',
        'dir' : './results/cifar10/pl',
        'lrate' : 0.00001,
        'epochs' : 1000,
        'batch_size' : 300,
        'patience' : 100,
        'steps_per_epoch' : 10,
        'use_image_augmentation' : False,
        'use_dae' : True,
    },
    'cifar10supervised' : {
        'method' : 'supervised',
        'test' : 'cifar10_experimental',
        'dir' : './results/cifar10/supervised',
        'lrate' : 0.0001,
        'epochs' : 1000,
        'batch_size' : 200,
        'patience' : 500,
        'steps_per_epoch' : 10,
        'use_image_augmentation' : True,
        'num_classes' : 10
    }
}

def execute_exp(args):

    if args['test'] in tests.tests:
        base_model, test = tests.tests[args['test']]
    else:
        print('error: test', args['test'], 'not found')
        return

    if args['method'] in models.models:
        # construct the model
        model = models.models[args['method']](base_model, args)
    else:
        print('error: model', args['method'], 'not found') 
        return

    # run the tests
    return test(model)

def save_results(model, results, out_dir, index):
    # save the output
    fname = out_dir + '/out_'+str(index)+'.pkl'

    with open(fname, 'wb') as f:
        pkl.dump(results, f)

    model.model.save(out_dir)

if __name__ == "__main__":

    # tensorflow stuff - limit ourselves to GPU memory

    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    import sys

    index = sys.argv[1]

    for arg in sys.argv[2:]:
        # try:
        print('\nrunning experiment:', arg, '\n')
        if arg in configs:
            config = configs[arg]
            print(config)
            model, results = execute_exp(config)

            save_results(model, results, config['dir'], index)
        else:
            print('error:', arg, 'not found')
        # except:
        #     print('some error occured')
        
        print('\n')