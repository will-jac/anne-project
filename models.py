from pseudo_labels import PseudoLabels

models = {
    'pseudo' : lambda args, in_size, out_size : PseudoLabels(
        in_size, args.hidden, out_size, args.lrate, 
        args.activation, args.out_activation, args.dropout, args.use_dae),
    'none' : lambda args, in_size, out_size: print('no model! args passed:', args)
}