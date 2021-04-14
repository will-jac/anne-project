import numpy as np
import pickle as pkl
import argparse

# ML methods
import models
import tests

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='ANNE', fromfile_prefix_chars='@')
    parser.add_argument('-method', type=str, default='pseudo', help='kernel method to use')
    parser.add_argument('-test', type=str, default='adult', help='test dataset to use')
    parser.add_argument('-dir', type=str, default='./results', help='location to store results')
    ## The following are parameters for the models. They're (mostly) optional, and may not be
    # used by all models
    parser.add_argument('-lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-hidden', nargs='+', type=int, default=[100, 10], help='number of hidden neurons'),
    parser.add_argument('-activation', type=str, default='relu', help='hidden activation function'),
    parser.add_argument('-out_activation', type=str, default='softmax', help='output activation function')
    parser.add_argument('-dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument('-use_dae', type=bool, default=True, help='to use DAE or not (PL)')

    
    return parser

def execute_exp(args):

    if args.method in models.models:
        model = models.models[args.method]
    else:
        print('error: model', args.method, 'not found') 
        return
    
    if args.test in tests.tests:
        test, n_ins, n_outs = tests.tests[args.test]
    else:
        print('error: test', args.test, 'not found')
        return

    # construct the model
    model = model(args, n_ins, n_outs)

    # run the tets
    results = test(model)

    # save the output
    fname = args.dir + '/' +  args.test + '_' + args.method  + '.out'

    with open(fname, 'wb') as f:
        out = {
            'args': args,
            'results': results
        }
        pkl.dump(out, f)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    execute_exp(args)
