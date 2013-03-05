import random
from collections import Counter

def k_fold_cross_validation(X, K):
    "Generates K (training, validation) pairs from the items in X."
    X = list(X) 
    random.shuffle(X)
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield k+1, training, validation

def filter_motifs(fold, n=2):
    "Filter motifs from the validation set that do not appear in the training set."
    k, training, validation = fold 
    label_fd = Counter(label for (_, labels, _) in training
                             for label in labels)
    training_labels = set(label for label, freq in label_fd.iteritems() if freq >= n)
    validation = [(source, labels & training_labels, text) for 
                  (source, labels, text) in validation if (labels & training_labels)]
    return k, training, validation