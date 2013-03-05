from collections import defaultdict

import numpy as np

import pylab as pl

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve

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
        assert isinstance(text, list)
        motif_ids = []
        for motif in motifs.split(','):
            if motif == 'DUMMY': continue
            if motif not in class_index:
                class_index.add(motif)
            motif_ids.append(class_index[motif])
        yield source, text, tuple(motif_ids)

def run(training, validation, k, config):

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

    labels, texts = zip(*texts_by_motifs.items())
    sources, labelings, tests = zip(*validation)
    labelings = [set(class_index[l] for l in ls) for ls in labelings]

    classifier = Pipeline([
                ('vectorizer', CountVectorizer(min_df=1, max_df=1.0, analyzer=lambda t: t)),
                ('tfidf', TfidfTransformer(norm='l2')),
                ('clf', MultinomialNB(alpha=0.000001))])

    classifier.fit(texts, labels)
    isError, OneError, nDocs = 0, 0, 0
    margins, AP = [], []
    predictions = classifier.predict_proba(tests)
    references = np.zeros((len(predictions), len(predictions[0])))
    for j, prediction in enumerate(predictions):
        nDocs += 1
        references[j][list(labelings[j])] = 1
        refs = np.zeros(len(prediction))
        refs[list(labelings[j])] = 1
        preds = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)
        refs = set(labelings[j])
        ap = average_precision(preds, refs)
        AP.append(ap)
        isError += is_error(ap)
        OneError += one_error(preds, refs)
        margins.append(margin(preds, refs))
    # if k is 0:
    #     precision, recall, _ = precision_recall_curve(references, predictions)
    #     for (p, r) in zip(precision, recall):
    #         print "%s\t%s" % (p, r)
    return isError, OneError, nDocs, margins, AP

