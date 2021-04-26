
from pseudo_labels import PseudoLabels
from temporal_ensembling import PiModel, TemporalEnsembling

# TODO: make this form of passing a model work for pseudo labels

models = {
    'pseudo' : lambda args, model : PseudoLabels(model, args.lrate, 
        args.activation, args.out_activation, args.dropout, args.use_dae),
    'pi' : lambda model, args : PiModel(model, 
            args.epochs, args.batch_size),
    'te' : lambda model, args : TemporalEnsembling(model, 
            args.epochs, args.batch_size),

    'none' : lambda model, args: print('no model! args passed:', args)
}