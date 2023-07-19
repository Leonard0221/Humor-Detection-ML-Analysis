import csv
from six.moves import cPickle as pickle
import numpy as np

x = []
with open('humorous_oneliners.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader: x.append(line)

with open('humorous_oneliners.pickle', 'w') as f:
    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
