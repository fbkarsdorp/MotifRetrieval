import csv
import logging
import sys

from collections import defaultdict
from time import time
from string import punctuation

import numpy as np

from sklearn.cross_validation import LeaveOneOut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import metrics


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

motifindex = defaultdict(list)
for i, row in enumerate(csv.reader(open('vvb-subset.csv'))):
    for motif in row[1].split():
        motifindex[motif].append(i)

vectorizer = TfidfVectorizer(min_df=1, analyzer=lambda t: (w for w in t.split() if w not in punctuation), norm="l2")
data = vectorizer.fit_transform(row[2] for row in csv.reader(open('vvb-subset.csv')))
feature_names = np.asarray(vectorizer.get_feature_names())

def benchmark(clf, X_train, y_train, X_test, y_test):
    # print 80 * '_'
    # print 'Training: '
    # print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    # print 'train time: %0.3fs' % train_time
    
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    # print 'test time:  %0.3f' % test_time
    
    macroscore = metrics.f1_score(y_test, pred, average='macro')
    # print 'f1-score (macro):   %0.3f' % macroscore
    
    microscore = metrics.f1_score(y_test, pred, average='micro')
    # print 'f1-score (micro):   %0.3f' % microscore
    
    clf_descr = str(clf).split('(')[0]
    return clf_descr, macroscore, microscore, train_time, test_time

mainresults = []
# set up a classifier for each motif (one-versus-all)
for motif, texts in motifindex.iteritems():
    results = []
    if len(texts) < 12: continue # skip motifs that occur only once
    targets = np.zeros(data.shape[0], dtype=int)
    targets[texts] = 1
    loo = LeaveOneOut(data.shape[0])
    t0 = time()
    for train_index, test_index in loo:
        # print 80 * '_'
        # print 'Motif: %s' % motif
        # print 'TRAIN: %s TEST: %s' % (train_index, test_index)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        results.append(benchmark(MultinomialNB(alpha=0.002), X_train, y_train, X_test, y_test))
        
    print 'SUMMARY', motif
    print 'Occurrences = %d, %0.3f' % (len(texts), 1 - float(len(texts)) / data.shape[0])
    print 'f1-score (macro):   %0.3f' % (sum(result[1] for result in results) / len(results))
    print 'f1-score (micro):   %0.3f' % (sum(result[2] for result in results) / len(results))


