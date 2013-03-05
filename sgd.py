import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import average_precision_score

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

def run(training, validation, k, config=None):
    isError, OneError, nDocs = 0, 0, 0
    margins, AP = [], []

    class_index = Index()
    traindocs, train_X, train_y = zip(*load_data(training, class_index))
    testdocs, test_X, test_y = zip(*load_data(validation, class_index))

    n_iter = np.ceil(10**6 / len(traindocs))

    clf = SGDClassifier(alpha=.000001, loss='log', n_iter=50, penalty='elasticnet')
    #clf = MultinomialNB(alpha=0.000001)

    classifier = Pipeline([
                ('vectorizer', CountVectorizer(min_df=1, max_df=1.0, analyzer=lambda t: t)),
                ('tfidf', TfidfTransformer(norm='l2')),
                ('clf', OneVsRestClassifier(clf, n_jobs=-1))])

    classifier.fit(train_X, train_y)
    predictions = classifier.predict_proba(test_X)
    for j, prediction in enumerate(predictions):
        nDocs += 1
        refs = np.zeros(len(prediction))
        refs[list(test_y[j])] = 1
        preds = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)
        refs = set(test_y[j])
        ap = average_precision(preds, refs)
        AP.append(ap)
        isError += is_error(ap)
        OneError += one_error(preds, refs)
        margins.append(margin(preds, refs))
    return isError, OneError, nDocs, margins, AP

