import csv
import re
import sys
from lxml import etree
from ucto import Tokenizer

tokenizer = Tokenizer('-L nl -n -Q')
data = etree.parse(sys.argv[1])
with open(sys.argv[1]+".csv", 'w') as output:
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    for row in data.getroot():
        _, idnumber, text, summary = list(row.iterchildren())
        motifs = set()
        if summary.text is None: continue
        for motiflist in re.findall('\[.*?\]', summary.text):
            for motif in re.findall('[^\s\[\],]+', motiflist):
                if not re.search('[0-9]', motif): continue
                motifs.add(motif)
        if motifs:
            print idnumber.text
            text = tokenizer.tokenize(text.text, verbose=False)
            writer.writerow([idnumber.text, r' '.join(motifs), ' '.join(text)])