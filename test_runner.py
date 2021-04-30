import numpy as np
import pickle as pkl
import argparse

# ML methods 
import models
import tests

from types import SimpleNamespace

configs = {
    'cifar10pi' :{
        'method' : 'pi',
        'test' : 'cifar10_experimental',
        'dir' : './results/cifar10/pi',
        'epochs' : 10000,
        'batch_size' : 300,
        'patience' : 500,
        'steps_per_epoch' : 10,
        'use_image_augmentation' : True,
    },
    'cifar10te' : {
        'method' : 'te',
        'test' : 'cifar10_experimental',
        'dir' : './results/cifar10/te',
        'epochs' : 10000,
        'batch_size' : 300,
        'patience' : 500,
        'steps_per_epoch' : 10,
        'use_image_augmentation' : True,
    },
    'cifar10pl' : {
        'method' : 'pl',
        'test' : 'cifar10_experimental',
        'dir' : './results/cifar10/pl',
        'lrate' : 0.00001,
        'epochs' : 10000,
        'batch_size' : 300,
        'patience' : 1000,
        'steps_per_epoch' : 10,
        'use_image_augmentation' : False,
        'use_dae' : False,
    },
    'cifar10supervised' : {
        'method' : 'supervised',
        'test' : 'cifar10_experimental',
        'dir' : './results/cifar10/supervised',
        'lrate' : 0.0001,
        'epochs' : 1000,
        'batch_size' : 100,
        'patience' : 500,
        'steps_per_epoch' : 2,
        'use_image_augmentation' : True,
    }
}

def execute_exp(run_config):

    args = SimpleNamespace(**run_config)

    if args.test in tests.tests:
        base_model, test = tests.tests[args.test]
    else:
        print('error: test', args.test, 'not found')
        return

    if args.method in models.models:
        # construct the model
        model = models.models[args.method](base_model, args)
    else:
        print('error: model', args.method, 'not found') 
        return

    # run the tests
    return test(model)

def save_results(model, results, out_dir):
    # save the output
    fname = out_dir + '/out.pkl'

    with open(fname, 'wb') as f:
        pkl.dump(results, f)

    model.model.save(out_dir)

if __name__ == "__main__":

    # tensorflow stuff - limit ourselves to GPU memory

    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    import sys

    if sys.argv[1] in configs:
        config = configs[sys.argv[1]]
        
        model, results = execute_exp(config)

        save_results(model, results, config['dir'])
    else:
        print('error:', sys.argv[1], 'not found')