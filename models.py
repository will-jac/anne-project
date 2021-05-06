
# from pseudo_labels import PseudoLabels
from model_runner import PiModel, TemporalEnsembling, PseudoLabels
from base_models import Supervised

# TODO: make this form of passing a model work for pseudo labels

models = {
    'pl' : lambda model, args : PseudoLabels(model, args),
    'pi' : lambda model, args : PiModel(model, args),
    'te' : lambda model, args : TemporalEnsembling(model, args),
    'supervised' : lambda model, args : Supervised(model, args),
    'none' : lambda model, args: print('no model! args passed:', args)
}