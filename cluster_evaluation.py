
from itertools import combinations
from collections import defaultdict

import numpy as np

import os
import gzip

from sklearn.cluster import DBSCAN, Ward
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import pairwise_distances

def read_topic_distribution(filename):
    features = []
    feature_vector = DictVectorizer(sparse=False)
    topic_vector = []
    topic_dists = defaultdict(list)
    for row in open(filename):
        try:
            topic, word, rank, score = row.strip().split(',')
        except ValueError:
            continue
        topic_dists[topic].append((word, float(score)))
    for motif, top_words in topic_dists.iteritems():
        topic_vector.append(motif)
        features.append(dict(top_words))
    return feature_vector, feature_vector.fit_transform(features), topic_vector

def read_term_topic_distribution(dirname):
    term_index = [line.strip() for line in open(os.path.join(dirname, '01000/term-index.txt'))]
    label_index = [line.strip() for line in open(os.path.join(dirname, '01000/label-index.txt'))]
    feature_vector = DictVectorizer(sparse=False)
    features = []
    labels = []
    for j,row in enumerate(gzip.open(os.path.join(dirname, '01000/topic-term-distributions.csv.gz'))):
        scores = [float(score) for score in row.split(',')]
        if sum(scores) > 0:
            features.append(dict((term_index[i], score) for i, score in enumerate(scores)))
            labels.append(label_index[j])
    return feature_vector, feature_vector.fit_transform(features), labels




def cluster_evaluation(D, y_true, n_clusters, eps=0.8, min_samples=10):
    ##############################################################################
    # Extract Y true
    labels_true = y_true

    ##############################################################################
    # transform distance matrix into a similarity matrix
    S = 1 - D 

    ##############################################################################
    # compute DBSCAN
    #db = DBSCAN(eps=eps, min_samples=min_samples).fit(S)
    db = Ward(n_clusters=n_clusters).fit(S)
    #core_samples = db.core_sample_indices_
    labels = db.labels_

    # number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print 'Number of clusters: %d' % n_clusters_
    print 'Homogeneity: %0.3f' % metrics.homogeneity_score(labels_true, labels)
    print 'Completeness: %0.3f' % metrics.completeness_score(labels_true, labels)
    print 'V-meassure: %0.3f' % metrics.v_measure_score(labels_true, labels)
    print 'Adjusted Rand Index: %0.3f' % metrics.adjusted_rand_score(labels_true, labels)
    print 'Adjusted Mutual Information: %0.3f' % metrics.adjusted_mutual_info_score(labels_true, labels)
    print 'Silhouette Coefficient: %0.3f' % metrics.silhouette_score(D, labels, metric='precomputed')

# Dutch max df = 0.25 -- larger vocabulary
# Frisian max df = 0.55 -- smaller vocbulary
vectorizer = CountVectorizer(min_df=1, max_df=0.25, analyzer=lambda text: text.split())
text_vector = []
motif_vector = defaultdict(list)
for i,row in enumerate(open('dutch-filtered.txt')):
    source, motifs, text = row.strip().split(',')
    motifs = motifs.split()
    for motif in motifs:
        motif_vector[motif].append(i)
    text_vector.append(text)

num_categories = len(set(m[0] for m in motif_vector))

data = vectorizer.fit_transform(text_vector).toarray()

motif_vector = motif_vector.items()
motif_separated_texts = np.array([data[indices].sum(axis=0) for motif,indices in motif_vector])
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf = tfidf_transformer.fit_transform(motif_separated_texts)
# dm = pairwise_distances(tfidf.toarray()[:,np.where(tfidf_transformer.idf_ > 1)[0]], metric='cosine', n_jobs=-1)
dm = pairwise_distances(tfidf.toarray(), metric='cosine', n_jobs=-1)

print 'TF.IDF topics'
print '-------------'
cluster_evaluation(dm, [m[0] for m,_ in motif_vector], num_categories, 0.95, 10)
print

tfidf = tfidf.toarray()
vocabulary = np.array(vectorizer.get_feature_names())
for i,(motif, _) in enumerate(motif_vector):
    vector = tfidf[i]
    topscore_indexes = np.argsort(tfidf[i])[-20:][::-1]
    print motif
    for word, score in zip(vocabulary[topscore_indexes], vector[topscore_indexes]):
        print score, word
    print

# FR python preprocessing.py --input frisian.txt --output frisian-filtered.txt --encoding utf-8 --strip-accents unicode --strip-punctuation --lowercase --min-df 2 --max-df 0.4 --word-length 2 --label-df 2 --labeled
# NL
f, fv, tv = read_term_topic_distribution('llda-vvb-0415cdeb-657-f1248203-f355fce4')
num_categories = len(set(m[0] for m in tv))
D = pairwise_distances(fv, metric='cosine', n_jobs=-1)
print 'LLDA topics'
print '-----------'
cluster_evaluation(D, [m[0] for m in tv], num_categories, 0.95, 10)

