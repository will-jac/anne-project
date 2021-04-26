from base_models import Scifar10Model

from pseudo_labels import PseudoLabels
from temporal_ensembling import PiModel, TemporalEnsembling

# TODO: make this form of passing a model work for pseudo labels

models = {
    'pseudo' : lambda args, in_size, out_size : PseudoLabels(
        in_size, args.hidden, out_size, args.lrate, 
        args.activation, args.out_activation, args.dropout, args.use_dae),
    'pi' : lambda args, model : PiModel(model, args.lrate, args.batch_size),
    'te' : lambda args, model : TemporalEnsembling(model, args.lrate, args.batch_size),

    'none' : lambda args, model: print('no model! args passed:', args)
}