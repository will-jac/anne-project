
from pseudo_labels import PseudoLabels
from temporal_ensembling import PiModel, TemporalEnsembling
from base_models import Supervised

# TODO: make this form of passing a model work for pseudo labels

models = {
    'pseudo' : lambda model, args : PseudoLabels(model, 
            lrate=args.lrate, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            use_dae=args.use_dae
        ),
    'pi' : lambda model, args : PiModel(model, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            use_image_augmentation=args.use_image_augmentation
        ),
    'te' : lambda model, args : TemporalEnsembling(model, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            use_image_augmentation=args.use_image_augmentation
        ),
    'supervised' : lambda model, args : Supervised(model, args),
    'none' : lambda model, args: print('no model! args passed:', args)
}