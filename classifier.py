from collections import defaultdict

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from metrics import average_precision, one_error, is_error, margin

class Index(object):
    def __init__(self, terms=[]):
        self._tokens = []
        self._types = {}
        self.update(terms)

    def add(self, term):
        if term not in self._types:
            self._tokens.append(term)
            self._types[term] = len(self._types)
        return self._types[term]

    def update(self, terms):
        for term in terms:
            self.add(term)

    def __len__(self):
        return len(self._types)

    def __contains__(self, term):
        return term in self._types

    def __getitem__(self, term):
        return self._types[term]

    def __repr__(self):
        return str(self._types)

def load_data(data, class_index):
    for (source, motifs, text) in data:
#        assert isinstance(text, list)
        motif_ids = []
        for motif in motifs:
            if motif == 'DUMMY': continue
            if motif not in class_index:
                class_index.add(motif)
            motif_ids.append(class_index[motif])
        yield source, text, tuple(motif_ids)

def construct_bigdocuments(training):
    class_index = Index()
    texts_by_motifs = defaultdict(list)
    motifs_in_docs = defaultdict(list)

    for i, (source, motifs, text) in enumerate(training):
        for motif in motifs:
            if motif != 'DUMMY':
                if motif not in class_index:
                    class_index.add(motif)
                motifs_in_docs[class_index[motif]].append(i)
                texts_by_motifs[class_index[motif]].extend(text)
    return zip(*texts_by_motifs.items()), class_index

def run(training, validation, k, config):

    norm = config.get('tfidf', 'norm')
    smooth_idf = config.getboolean('tfidf', 'smooth_idf')

    bigdoc = config.getboolean('NB', 'bigdoc')
    clf = config.get('system', 'system')
    if clf == 'NB':
        clf = MultinomialNB(alpha=config.getfloat('NB', 'alpha'))
        if not bigdoc:
            clf = OneVsRestClassifier(clf, n_jobs=-1)
    elif clf == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=10, weights='distance')
        if not bigdoc:
            clf = OneVsRestClassifier(clf)
    elif clf == 'SVC':
        clf = LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-3)
        if not bigdoc:
            clf = OneVsRestClassifier(clf)
    elif clf == 'dtree':
        clf = DecisionTreeClassifier()
    else:
        clf = OneVsRestClassifier(
            SGDClassifier(alpha=config.getfloat('sgd', 'alpha'),
                          loss=config.get('sgd', 'loss'),
                          n_iter=config.getint('sgd', 'iterations'),
                          penalty=config.get('sgd', 'penalty')), n_jobs=-1)

    classifier = Pipeline([
        ('vectorizer', CountVectorizer(min_df=1, max_df=1, analyzer=lambda t: t)),
        ('tfidf', TfidfTransformer(norm=norm, smooth_idf=smooth_idf)),
        ('clf', clf)])

    if bigdoc:
        (train_y, train_X), class_index = construct_bigdocuments(training)
        _, test_y, test_X = zip(*validation)
        test_y = [set(class_index[l] for l in ls) for ls in test_y]
    else:
        class_index = Index()
        _, train_X, train_y = zip(*load_data(training, class_index))
        _, test_X, test_y = zip(*load_data(validation, class_index))

    classifier.fit(train_X, train_y)
    isError, OneError, nDocs = 0, 0, 0
    margins, AP = [], []
    predictions = classifier.predict_proba(test_X)
    for j, prediction in enumerate(predictions):
        nDocs += 1
        preds = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)
        refs = test_y[j]
        ap = average_precision(preds, refs)
        AP.append(ap)
        isError += is_error(ap)
        OneError += one_error(preds, refs)
        margins.append(margin(preds, refs))
    return isError, OneError, nDocs, margins, AP

