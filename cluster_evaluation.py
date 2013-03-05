
from itertools import combinations
from collections import defaultdict

import numpy as np

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

def cluster_evaluation(D, y_true, eps=0.8, min_samples=10):
    ##############################################################################
    # Extract Y true
    labels_true = y_true

    ##############################################################################
    # transform distance matrix into a similarity matrix
    S = 1 - D 

    ##############################################################################
    # compute DBSCAN
    #db = DBSCAN(eps=eps, min_samples=min_samples).fit(S)
    db = Ward(n_clusters=26).fit(S)
    #core_samples = db.core_sample_indices_
    labels = db.labels_

    # number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print 'Estimated number of clusters: %d' % n_clusters_
    print 'Homogeneity: %0.3f' % metrics.homogeneity_score(labels_true, labels)
    print 'Completeness: %0.3f' % metrics.completeness_score(labels_true, labels)
    print 'V-meassure: %0.3f' % metrics.v_measure_score(labels_true, labels)
    print 'Adjusted Rand Index: %0.3f' % metrics.adjusted_rand_score(labels_true, labels)
    print 'Adjusted Mutual Information: %0.3f' % metrics.adjusted_mutual_info_score(labels_true, labels)
    print 'Silhouette Coefficient: %0.3f' % metrics.silhouette_score(D, labels, metric='precomputed')


vectorizer = CountVectorizer(min_df=1, analyzer=lambda text: text.split())
text_vector = []
motif_vector = defaultdict(list)
for i,row in enumerate(open(
        '../LDA/data/vvb-frysian-no-tmi-min-df_4-max-df_1-word-length_3-label-df_2.txt-training.txt')):
    source, motifs, text = row.strip().split('\t')
    motifs = motifs.split(',')
    for motif in motifs:
        motif_vector[motif].append(i)
    text_vector.append(text)

data = vectorizer.fit_transform(text_vector).toarray()

motif_vector = motif_vector.items()
motif_separated_texts = np.array([data[indices].sum(axis=0) for motif,indices in motif_vector])
tfidf = TfidfTransformer(norm="l2").fit_transform(motif_separated_texts)
dm = pairwise_distances(tfidf.toarray(), metric='cosine', n_jobs=-1)

print 'TF.IDF topics'
print '-------------'
cluster_evaluation(dm, [m[0] for m,_ in motif_vector], 0.95, 10)
print

f, fv, tv = read_topic_distribution('../llda/llda-vvb-106bc98b-462-fab2d516-bd05b57a/top-terms.csv')
D = pairwise_distances(fv, metric='cosine', n_jobs=-1)
print 'LLDA topics'
print '-----------'
cluster_evaluation(D, [m[0] for m in tv], 0.95, 10)

