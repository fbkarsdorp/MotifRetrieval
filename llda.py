import csv
import os
import subprocess
import sys

from collections import defaultdict

from metrics import average_precision, one_error, is_error, margin


def run(training, validation, k, config):
    ground_truth = {}
    ROOTDIR = config.get('filepaths', 'corpus')
    alpha, beta = config.get('llda', 'alpha'), config.get('llda', 'beta')
    iterations = config.get('llda', 'iterations')

    with open(ROOTDIR + 'training-%s.tmp' % k, 'w') as training_out:
        writer = csv.writer(training_out, quoting=csv.QUOTE_MINIMAL)
        for (source, motifs, text) in training:
            motifs = r' '.join(motif for motif in motifs if motif != 'DUMMY')
            writer.writerow([source, motifs, ' '.join(text)])

    with open(ROOTDIR + 'testing-%s.tmp' % k, 'w') as testing_out:
        writer = csv.writer(testing_out, quoting=csv.QUOTE_MINIMAL)
        for (source, motifs, text) in validation:
            ground_truth[source] = motifs
            writer.writerow([source, r' '.join(motif), ' '.join(text)])
    
    # train LLDA
    with open(os.devnull, 'w') as null:
        subprocess.call('java -Xmx2000mb -jar tmt-0.4.0.jar llda-train.scala %s %s %s %s' %
            (ROOTDIR + 'training-%s.tmp' % k, alpha, beta, iterations),
            stdout=null, stderr=null, shell=True)
    # retrieve the model path
    modelpath = open(ROOTDIR + 'training-%s.tmp.config' % k).read().strip()
    # preform inference on led-out dataset using trained model
    with open(os.devnull, 'w') as null:
        subprocess.call('java -Xmx2000mb -jar tmt-0.4.0.jar llda-test.scala %s %s' %
            (modelpath, (ROOTDIR + 'testing-%s.tmp' % k)),
            stdout=sys.stdout, stderr=sys.stderr, shell=True)

    # evaluation starts here!
    isError, oneError, nDocs = 0, 0, 0
    AP, margins = [], []
    label_file = '/%05d/label-index.txt' % config.getint('llda', 'iterations')
    topicIndex = [topic.strip() for topic in open(modelpath + label_file)]
    reader = csv.reader(open(modelpath + '/testing-%s.tmp-document-topic-distributuions.csv' % k))
    for row in reader:
        nDocs += 1
        idnumber, topics = row[0], [float(score) for score in row[1:]]
        topics = sorted([(topicIndex[i], score) for i, score in enumerate(topics)],
                        key=lambda i: i[1], reverse=True)
        preds = [topic for topic, _ in topics if topic != 'DUMMY']
        refs = ground_truth[idnumber]
        ap = average_precision(preds, refs)
        isError += is_error(ap)
        oneError += one_error(preds, refs)
        margins.append(margin(preds, refs))
        AP.append(ap)
    return isError, oneError, nDocs, margins, AP


