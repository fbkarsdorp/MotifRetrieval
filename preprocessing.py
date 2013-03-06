import argparse
import codecs
import cPickle
import csv
import logging
import os
import string
import sys
import time
import uuid
import unicodedata

from collections import Counter, defaultdict
from itertools import ifilter

from TMI import TMI


def identity(x): return x

# from http://docs.python.org/library/csv.html#csv-examples
def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data),
                            dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

def cleanfile(filename, processor, label_df=False, tmi=None, encoding='utf-8'):
    with codecs.open(filename, encoding=encoding) as f:
        reader = unicode_csv_reader(f)
        sources, labels, texts, observedlabels = [], [], [], []
        if label_df:
            for (source, labeling, text) in reader:
                labeling = clean_labels(labeling)
                observedlabels.append(labeling)
                if tmi is not None:
                    labeling = add_parent_labels(labeling, tmi)
                else:
                    labeling = ' '.join(labeling)
                if not labeling:
                    continue # TODO fix this!
                sources.append(source)
                labels.append(labeling)
                texts.append(text)
            # make a string representation of the labels, here...
            labels = [l for l in termCountFilter(labels, 1.0, label_df)]
            if tmi is not None:
                for topics in labels:
                    assert 'ROOT' in topics, topics
            if tmi is not None:
                unarylabels = filter_unary_parent_labels(observedlabels, labels, tmi)
                #print unarylabels
                labels = [set(l)-unarylabels for l in labels]
            #labels = [','.join(sorted(l)) for l in labels]
            for source,label,text in zip(sources, labels, processor(texts)):
                if label:
                    yield (source, label, text)
        else:
            for (source, text) in reader:
                sources.append(source)
                texts.append(text)
            for source,text in zip(sources, processor(texts)):
                yield (source, text)

def strip_punctuation(s):
    """Return the text without (hopefully...) any punctuation.
    >>> strip_punctuation('this.is?as!funny@word') == 'thisisasfunnyword'
    """
    if isinstance(s, unicode):
        return ''.join(ch for ch in s if unicodedata.category(ch)[0] not in ('P', 'S'))
    return s.translate(string.maketrans("",""), string.punctuation)

def strip_accents_unicode(s):
    "Return the text with all unicode accents stripped."
    return u''.join(c for c in unicodedata.normalize('NFKD', s)
                    if not unicodedata.combining(c))

def strip_accents_ascii(s):
    "Return the text with all ACCII accents stripped."
    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')

def clean_labels(labels):
    return set(l[:-1] if l.endswith('.') else l for l in labels.split())

def add_parent_labels(labels, tmi):
    labels = set(label for label in labels if label in tmi)
    parentlabels = set(parent for label in labels for parent in tmi.parents(label))
    if labels: assert 'ROOT' in parentlabels, labels
    return ' '.join(labels | parentlabels)

def filter_unary_parent_labels(documentlabels, parentlabels, tmi):
    observedlabels = set(label for labels in documentlabels for label in labels)
    labeltree = defaultdict(set)
    for labels in parentlabels:
        for label in labels:
            for child in tmi.predecessors(label):
                if child in labels:
                    labeltree[label].add(child)
    # return set()
    return set(l for l,c in labeltree.iteritems() if len(c) is 1 and l not in observedlabels)

def termCountFilter(documents, max_df, min_df, split=string.split, stoplist=set(),
                    word_clean_fn=identity, doc_clean_fn=identity, min_word_len=0, join=False):
    term_counts = Counter()
    term_counts_per_doc = []
    document_counts = Counter()
    cleaned_documents = []

    for document in documents:
        doc = [word_clean_fn(word) for word in split(doc_clean_fn(document))]
        term_count_current = Counter(doc)
        cleaned_documents.append(doc)
        term_counts.update(term_count_current)
        document_counts.update(term_count_current.iterkeys())
        term_counts_per_doc.append(term_count_current)

    n_doc = len(term_counts_per_doc)
    max_doc_count = max_df * n_doc
    min_doc_count = min_df

    # make a list of stopwords
    if stoplist:
        stoplist = stoplist
    elif max_doc_count < n_doc or min_doc_count > 1:
        stoplist = set(t for t, dc in document_counts.iteritems()
                          if dc > max_doc_count or dc < min_doc_count)
    else: 
        stoplist = set()
    # which words do we include?
    terms = set(w for w in term_counts if len(w) >= min_word_len) - stoplist
    for document in cleaned_documents:
        doc = [term for term in document if term in terms]
        if join:
            yield ' '.join(doc)
        else:
            yield doc


def preprocess(documents, encoding='utf-8', strip_accents=None, strip_punct=True,
               stopwords=set(), lowercase=True, min_n=None,
               max_n=None, max_df=1.0, min_df=2, min_word_len=3, join=False):
    "Apply some preprocessing steps to the text."
    strip_accents = (strip_accents_ascii if strip_accents == 'ascii' else
                     strip_accents_unicode if strip_accents == 'unicode' else
                     identity)
    strip_punct = strip_punctuation if strip_punct else identity
    lowercase = string.lower if lowercase else identity
    return termCountFilter(documents, max_df, min_df, string.split, stopwords,
                           lambda w: lowercase(strip_accents(w)), strip_punct,
                           min_word_len, join)

def main(args):
    with open(args.inputfile + '-' + str(uuid.uuid1()) + '.log', 'w') as logfile:
        logfile.write(
            ('TIME: %s\nINPUT: %s\nOUTPUT: %s\nLABELS: %s\nSTOPWORDS: %s\n' + 
            'STRIP-ACCENTS: %s\nSTRIP-PUNCTUATION: %s\nLOWERCASE: %s\n' +
            'MIN-DF: %s\nMAX-DF: %s\nWORD-LENGTH: %s\nLABEL-DF: %s\n' +
            'TMI: %s') % (
                time.ctime(), args.inputfile, args.outputfile, args.labels,
                args.stopwords, args.strip_accents, args.strip_punct,
                args.lowercase, args.min_df, args.max_df, args.min_word_len,
                args.label_df, args.add_tmi))
    stopwords = set() if args.stopwords is None else set(
        line.strip() for line in open(args.stopwords))
    if args.add_tmi:
        with open('../data/tmi.cPickle') as inf:
            tmi = cPickle.load(inf)
    else:
        tmi = None
    preprocessor = lambda t: preprocess(t, encoding=args.encoding, 
                                        strip_accents=args.strip_accents, 
                                        strip_punct=args.strip_punct,
                                        stopwords=stopwords, 
                                        lowercase=args.lowercase,
                                        max_df=args.max_df, 
                                        min_df=args.min_df, 
                                        min_word_len=args.min_word_len, 
                                        join=True)
    writeformat = '%s\t%s\t%s\n' if args.labels else '%s\t%s\n'
    if not isinstance(args.outputfile, file):
        outfile = open(args.outputfile, 'w')
    else:
        outfile = args.outputfile
    for document in cleanfile(args.inputfile, preprocessor, 
                              label_df=args.label_df, tmi=tmi,
                              encoding=args.encoding):
        try:
            outfile.write(writeformat % document)
        except UnicodeEncodeError:
            print "Unicode error in %r" % document[-1]
            raise UnicodeEncodeError
    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = "preprocessing", 
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = "")
    parser.add_argument('--input', dest='inputfile', type=str, required=True,
                        help = 'Specify the filename that needs to be processed.')
    parser.add_argument('--output', dest='outputfile', type = argparse.FileType('w'),
                        required=False, default=sys.stdout,
                        help = 'filename for output file.')
    parser.add_argument('--labeled', dest='labels', action='store_true', default=False,
                        help='Does the file contain observed labels for each document?')
    parser.add_argument('--encoding', dest='encoding', type=str, default='utf-8')
    parser.add_argument('--stopwords', dest='stopwords', type=str, required=False,
                        help='filename for a one-per-line stopword list')
    parser.add_argument('--strip-accents', dest='strip_accents', type=str,
                        choices = ['ascii', 'unicode'], required=False, default='unicode',
                        help = 'Remove all accents from the text?')
    parser.add_argument('--strip-punctuation', dest='strip_punct', action='store_true',
                        default=True, help = 'Remove all punctuation?')
    parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                        default=True, help = 'Convert words to lowercase?')
    parser.add_argument('--min-df', dest='min_df', type=int, default=2,
                        help='The minimum document frequency of a word.')
    parser.add_argument('--max-df', dest='max_df', type=float, default=1.0,
                        help='The maximum document frequency represented by a proportion parameter (Float in range [0.0, 1.0])')
    parser.add_argument('--word-length', dest='min_word_len', default=3, type=int,
                        help='The minumum length of a word.')
    parser.add_argument('--label-df', dest='label_df', type=int, default=1,
                        help='The minimum document label frequency.')
    parser.add_argument('--tmi', dest='add_tmi', action='store_true', default=False,
                        help='Add the parents in TMI to the set of observed labels?')
    args = parser.parse_args()
    
    main(args)
    


