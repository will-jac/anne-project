
# from pseudo_labels import PseudoLabels
from model_runner import PiModel, TemporalEnsembling, PseudoLabels
from base_models import Supervised

# TODO: make this form of passing a model work for pseudo labels

models = {
    'pl' : lambda model, args : PseudoLabels(model, 
            lrate=args.lrate, 
            epochs=args.epochs,
            min_labeled_per_epoch=args.min_labeled_per_epoch,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            patience=args.patience,
            use_image_augmentation=args.use_image_augmentation
        ),
    'pi' : lambda model, args : PiModel(model, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            min_labeled_per_epoch=args.min_labeled_per_epoch,
            steps_per_epoch=args.steps_per_epoch,
            patience=args.patience,
            use_image_augmentation=args.use_image_augmentation
        ),
    'te' : lambda model, args : TemporalEnsembling(model, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            min_labeled_per_epoch=args.min_labeled_per_epoch,
            patience=args.patience,
            use_image_augmentation=args.use_image_augmentation
        ),
    'supervised' : lambda model, args : Supervised(model, args),
    'none' : lambda model, args: print('no model! args passed:', args)
}