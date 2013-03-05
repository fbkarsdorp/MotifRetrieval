
from collections import defaultdict

from BM25 import Indexer
from metrics import average_precision, one_error, is_error, margin


def run(training, validation, k, config):
    texts_by_motifs = defaultdict(list)
    motifs_in_docs = defaultdict(list)

    # construct the bigdocuments
    for i, (source, motifs, text) in enumerate(training):
        for motif in motifs:
            if motif != 'DUMMY':
                motifs_in_docs[motif].append(i)
                texts_by_motifs[motif].extend(text)

    labels, texts = zip(*texts_by_motifs.items())
    indexer = Indexer()
    for label, text in zip(labels, texts):
        indexer.add(text, label)

    isError, OneError, nDocs = 0, 0, 0
    margins, AP = [], []
    for j, (source, motifs, text) in enumerate(validation):
        nDocs += 1
        scores = list(indexer.predict_proba(
            text, config.getfloat('bm25', 'k1'), config.getfloat('bm25', 'b')))
        preds = sorted(scores, key=lambda i: i[1], reverse=True)
        preds = [label for label,score in preds]
        refs = set(motifs)
        ap = average_precision(preds, refs)
        AP.append(ap)
        isError += is_error(ap)
        OneError += one_error(preds, refs)
        margins.append(margin(preds, refs))
    return isError, OneError, nDocs, margins, AP
