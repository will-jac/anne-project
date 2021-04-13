
# ANNE Large Project

## Notes for Clark

1. I created a util.py file that I intend to put a lot of things into that aren't directly NN construction, but related.
One really important thing in there is the data namedtuple, which is a cool python thing that lets you have a tuple
(that you can still access by data[0], data[1], in the usual way) that has names (eg data.X data.y). This will be
very useful for making all of our wrapper algorithms so you don't have to remember the order of things, and because
our data consists of labeled + unlabeled data, so having easy access to this is important.
2. 4/12 update: I added a running utility, using argparse. Just put your model in models.py and any tests in tests.py and it should work - you may need to add additional parameters though.
3. 4/12 update: I changed the Data tuple to provide X, y, U directly, because it's just easier to work with

## TODO

* more robust model saving / history
* more tests!
