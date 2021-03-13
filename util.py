
from collections import namedtuple

Data = namedtuple('Data', 'Labeled Unlabeled')

LabeledData = namedtuple('LabeledData', 'X y')

UnlabeledData = namedtuple('UnlabeledData', 'X')

## see pseudo-labels for an example of usage