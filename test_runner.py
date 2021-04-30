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
        'epochs' : 500,
        'batch_size' : 100,
        'steps_per_epoch' : 2,
        'use_image_augmentation' : True,
    },
    'cifar10te' : {
        'method' : 'te',
        'test' : 'cifar10',
        'dir' : './results/cifar10/te',
        'epochs' : 100,
        'batch_size' : 100,
        'steps_per_epoch' : 2,
        'use_image_augmentation' : True,
    },
    'cifar10pl' : {
        'method' : 'pl',
        'test' : 'cifar10',
        'dir' : './results/cifar10/pl',
        'lrate' : 0.0001,
        'epochs' : 100,
        'batch_size' : 100,
        'steps_per_epoch' : 2,
        'use_image_augmentation' : True,
        'use_dae' : False
    },
    'cifar10supervised' : {
        'method' : 'supervised',
        'test' : 'cifar10',
        'dir' : './results/cifar10/supervised',
        'lrate' : 0.0001,
        'epochs' : 100,
        'batch_size' : 100,
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
    results = test(model)
    return model, results

def save_results(model, results):
    # save the output
    fname = args.dir + '/' +  args.test + '_' + args.method  + '.out'

    with open(fname, 'wb') as f:
        out = {
            'args': args,
            'results': results
        }
        pkl.dump(out, f)

if __name__ == "__main__":

    # tensorflow stuff - limit ourselves to GPU memory

    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    import sys

    if sys.argv[1] in configs:
        config = configs[sys.argv[1]]
        
        model, results = execute_exp(config)
        save_results(model, results)
    else:
        print('error:', sys.argv[1], 'not found')