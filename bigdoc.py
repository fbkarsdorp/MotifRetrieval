import os
import sys

from collections import defaultdict

if sys.argv[1] == 'bigdoc':
    motifs_in_doc = defaultdict(list)

    for i, row in enumerate(open(sys.argv[2])):
        source, motifs, text = row.strip().split('\t')
        motifs = motifs.split(',')
        for motif in motifs:
            if motif == 'DUMMY': continue
            motifs_in_doc[motif].extend(text.split())

    for motif, text in motifs_in_doc.iteritems():
        with open(os.path.join(sys.argv[3], motif), 'w') as out:
            out.write('<DOC><DOCNO>%s</DOCNO>%s</DOC>' % (motif, ' '.join(text)))

else:
    with open(sys.argv[3], 'w') as out:
        for i, row in enumerate(open(sys.argv[2])):
            source, motifs, text = row.strip().split('\t')
            out.write('<TOP>\n<NUM>%s</NUM>\n<ID>%s</ID>\n<TEXT> %s</TEXT>\n</TOP>\n\n' % (
                i, source, text))
    allmotifs = set(os.listdir(sys.argv[4]))
    with open(sys.argv[3]+'.rels', 'w') as out:
        for i, row in enumerate(open(sys.argv[2])):
            source, motifs, text = row.strip().split('\t')
            motifs = motifs.split(',')
            for motif in allmotifs:
                if motif == '.DS_Store': continue
                if motif == 'DUMMY': continue
                out.write("%s %s %s %s\n" % (i, 0, motif, (1 if motif in motifs else 0)))
        