import argparse
import codecs
import csv
import os
import subprocess
import sys

from collections import defaultdict
from ConfigParser import SafeConfigParser

from preprocessing import cleanfile, clean_labels, preprocess, unicode_csv_reader
from crossvalidation import filter_motifs, k_fold_cross_validation
from metrics import average_precision, one_error, is_error, margin

import llda
import MR
import classifier

# argparser = argparse.ArgumentParser()
# argparser.add_argument(
#     '-c', '--config', dest='config', help='Configuration file for experiment')
# argparser.add_argument(
#     '-s', '--system', dest='system', help='System to use (llda, MR, SVM).')
# args = argparser.parse_args()

def main(parameters):
    config = SafeConfigParser()
    config.read(parameters)

    ROOTDIR = config.get('filepaths', 'corpus')

    if len(os.listdir(ROOTDIR)) < 2:
        documents = []
        with codecs.open(config.get('filepaths', 'basefile'), encoding='utf-8') as f:
            for (source, labels, text) in unicode_csv_reader(f):
                labels = clean_labels(labels)
                documents.append((source, labels, text))
        for fold in k_fold_cross_validation(documents, 10):
            print fold
            fold, training, validation = filter_motifs(fold)
            with open(ROOTDIR + 'fold-%s.training.txt' % fold, 'w') as out:
                writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
                for (source, motifs, text) in training:
                    #motifs = [motif for motif in motifs is motif != 'DUMMY']
                    writer.writerow([source, ' '.join(motifs).encode('utf-8'), text.encode('utf-8')])
            with open(ROOTDIR + 'fold-%s.validation.txt' % fold, 'w') as out:
                writer = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
                for (source, motifs, text) in validation:
                    writer.writerow([source, ' '.join(motifs).encode('utf-8'), text.encode('utf-8')])

    training_preprocessor = lambda t: preprocess(
        t, encoding=config.get('preprocessing', 'encoding'),
        strip_accents = 'unicode' if config.getboolean('preprocessing', 'strip-accents') else None,
        strip_punct = config.getboolean('preprocessing', 'strip-punctuation'),
        lowercase = config.getboolean('preprocessing', 'lowercase'),
        max_df = config.getfloat('preprocessing', 'maximum-document-frequency'),
        min_df = config.getint('preprocessing', 'minimum-document-frequency'),
        min_word_len = config.getint('preprocessing', 'minimum-word-length'))

    validation_preprocessor = lambda t: preprocess(
        t, encoding=config.get('preprocessing', 'encoding'),
        strip_accents = 'unicode' if config.getboolean('preprocessing', 'strip-accents') else None,
        strip_punct = config.getboolean('preprocessing', 'strip-punctuation'),
        lowercase = config.getboolean('preprocessing', 'lowercase'),
        max_df = 1.0,
        min_df = 1.0,
        min_word_len = config.getint('preprocessing', 'minimum-word-length'))

    documents = defaultdict(list)
    for document in os.listdir(ROOTDIR):
        if not document.startswith('.') and document.startswith('fold'):
            documents[document.split('.')[0]].append(document)

    globalAP = []
    globalMargin = []
    globalOneError = []
    globalIsError = []

    system = config.get('system', 'system')
    if system == 'llda':
        system = llda
    elif system.upper() in ('SGD', 'SVC', 'KNN', 'NB'):
        system = classifier
    elif system == 'BM25':
        system = MR
    else:
        raise ValueError("Unsupported system choice: %s" % system)

    for k, (fold, (training_docs, test_docs)) in enumerate(documents.iteritems()):
        assert 'training' in training_docs and 'validation' in test_docs
        training = list(cleanfile(ROOTDIR + training_docs, training_preprocessor, label_df=1))
        validation = list(cleanfile(ROOTDIR + test_docs, validation_preprocessor, label_df=1))
        isError, oneError, nDocs, margins, AP = system.run(training, validation, k, config)
        isError = isError / float(nDocs)
        oneError = oneError / float(nDocs)
        margins = sum(margins) / float(nDocs)
        AP = sum(AP) / float(nDocs)
        globalIsError.append(isError)
        globalOneError.append(oneError)
        globalMargin.append(margins)
        globalAP.append(AP)

        print 'Fold:', k
        print '-' * 80
        print 'Num training docs:', len(training)
        print 'Num validation docs:', len(validation)
        print 'Average Precision:', AP
        print 'Is Error:', isError
        print 'One Error:', oneError
        print 'Margin:', margins
        print '-' * 80
    output_dir = os.path.join('Data', sys.argv[-1])
    with open(os.path.join(output_dir, 'output.txt'), 'w') as out:
        out.write('Average Precision: %f\n' % (sum(globalAP) / len(globalAP)))
        out.write('Average One Error: %f\n' % (sum(globalOneError) / len(globalOneError)))
        out.write('Average Is Error: %f\n' % (sum(globalIsError) / len(globalIsError)))
        out.write('Average Margin: %f\n' % (sum(globalMargin) / len(globalMargin)))

    print 'AVERAGE AP:', sum(globalAP) / len(globalAP)
    print 'AVERAGE ONE ERROR:', sum(globalOneError) / len(globalOneError)
    print 'AVERAGE IS ERROR:', sum(globalIsError) / len(globalIsError)
    print 'AVERAGE MARGIN:', sum(globalMargin) / len(globalMargin)

main(sys.argv[1])

