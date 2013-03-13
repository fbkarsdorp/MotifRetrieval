import math
from collections import defaultdict, Counter

class Indexer(object):
    """Simple Indexer, not very general and aimed at small scale retrieval 
    experiments"""
    def __init__(self):
        self.N = 0
        self.document_lengths = {}
        self._tf = defaultdict(lambda: defaultdict(int))
        self._df = defaultdict(int)
        self.idf_ = {}
        self.scores_ = {}
        self.avg_len = None
        self.documents = []

    def __repr__(self):
        return '<Indexer(N=%d)>' % self.N

    def tf(self, term, document):
        "Return the frequency of a term in some document."
        return self._tf.get(term, {}).get(document, 0)

    def df(self, term):
        "Return the document frequency of a term."
        return self._df.get(term, 0)

    def add(self, document, id=None):
        "Add a document to the index. ID is optional."
        self.N += 1
        if id is None: id = self.N
        assert id not in self.document_lengths, 'Document %s is already indexed' % id
        self.document_lengths[id] = len(document)
        self.documents.append(id)
        for term in document:
            if self.tf(term, id) is 0:
                self._df[term] += 1
            self._tf[term][id] += 1

    def fit(self, documents, ids=None):
        "Add multiple documents at once to the index."
        if ids is None: ids = range(len(documents))
        for document, id in zip(documents, ids):
            self.add(document, id)
        self.avg_len = sum(self.document_lengths.values()) / float(self.N)

    def idf(self, term):
        "Compute the inverse document frequency of a term."
        try:
            return self.idf_[term]
        except KeyError:
            self.idf_[term] = idf = math.log(
                (self.N - self.df(term) + 0.5) / (self.df(term) + 0.5))
            return idf

    def scores(self, query, document, k1=1.2, b=0.75, filter=True):
        for term in query:
            # try: # check whether we computed this already
                # s = self.scores_[term, document]
            # except KeyError:
            idf = self.idf(term)
            tf = self.tf(term, document)
            l = self.document_lengths[document]
            s = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * l / self.avg_len))
                # self.scores_[term, document] = s
            yield 0.0 if filter and s < 0.0 else s

    def BM25(self, query, document, k1=1.2, b=0.75, filter=True):
        "Compute the Okkapi BM25 score for one document, given a query."
        score = 0.0
        for term in query:
            try: # check whether we computed this already
                s = self.scores_[term, document]
            except KeyError:
                idf = self.idf(term)
                tf = self.tf(term, document)
                l = self.document_lengths[document]
                s = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * l / self.avg_len))
                self.scores_[term, document] = s 
            score += 0.0 if filter and s <= 0.0 else s
        return score

    def predict_proba(self, query, k1=1.2, b=0.75, filter=True):
        "Compute the Okkapi BM25 score for all documents, given a query."
        if self.avg_len is None: 
            self.avg_len = sum(self.document_lengths.values()) / float(self.N)
        for document in self.documents:
            yield document, self.BM25(query, document, k1, b, filter)

    def nbest_BM25(self, query, n=50):
        "Return the nbest matching documents given a query."
        scores = sorted(self.predict_proba(query), key=lambda s: s[1], reverse=True)
        return scores[:n]




