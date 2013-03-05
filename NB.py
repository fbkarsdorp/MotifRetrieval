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
from sgd import Index, load_data

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
    smooth_idf = config.getBoolean('tfidf', 'smooth_idf')

    bigdoc = False
    clf = config.get('system', 'system')
    if clf == 'NB':
        alpha=config.getFloat('NB', 'alpha')
        if config.getBoolean('NB', 'bigdoc'):
            bigdoc = True
            clf = MultinomialNB(alpha=alpha)
        else:
            clf = OneVsRestClassifier(BernoulliNB(alpha=alpha))
    else:
        clf = SGDClassifier(alpha=config.getFloat('sgd', 'alpha'),
                            loss=config.get('sgd', 'loss'),
                            n_iter=config.getInt('sgd', 'iterations'),
                            penalty=config.get('sgd', 'penalty'))

    classifier = Pipeline([
        ('vectorizer', CountVectorizer(min_df=1, max_df=1.0, analyzer=lambda t: t)),
        ('tfidf', TfidfTransformer(norm=norm, smooth_idf=smooth_idf)),
        ('clf', clf)])

    if bigdoc:
        (train_y, train_X), class_index = construct_bigdocuments(training)
        _, test_y, test_X = zip(*validation)
        test_y = [tuple(class_index[l] for l in ls) for ls in test_y]
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
        refs = set(labelings[j])
        ap = average_precision(preds, refs)
        AP.append(ap)
        isError += is_error(ap)
        OneError += one_error(preds, refs)
        margins.append(margin(preds, refs))
    return isError, OneError, nDocs, margins, AP

