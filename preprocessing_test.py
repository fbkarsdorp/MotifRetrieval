# coding: utf-8

# testing the preprocessing script, because a lot can go wrong...

import unittest
from preprocessing import *

class PreprocessingTest(unittest.TestCase):

    def setUp(self):
        self.file = "test.csv"      
        self.texts = []
        self.sources = []
        self.labels = []
        with codecs.open(self.file, encoding='utf-8') as f:
            for source, labels, text in unicode_csv_reader(f):
                self.sources.append(source)
                self.labels.append(labels)
                self.texts.append(text)

    def test_strip_punctuation_1(self):
        self.assertEqual(u'%s' % strip_punctuation(self.texts[0]), 
            u"Dit is tekst 1  Die slechts kort is met hier ú een accent en daar  een gek teken")
    
    def test_strip_punctuation_2(self):
        self.assertEqual(strip_punctuation(self.texts[1]),
            u"Dit is tekst 2  Ook hier  gekke tekens  leestekens  en woorden ")

    def test_strip_punctuation_3(self):
        self.assertEqual(strip_punctuation(self.texts[2]),
            u"Dit is de laatste tekst  Met wederom een aantal     tekens niet om te schelden hoor en unicode tekens zoals åü en ")

    def test_strip_accents(self):
        self.assertEqual(strip_accents_unicode(self.texts[0]),
            u"Dit is tekst 1 . Die slechts kort is met hier u een accent en daar ¬ een gek teken.")

    def test_full_cleanup(self):
        newtexts = []
        for text in self.texts:
            newtexts.append([string.lower(strip_accents_unicode(w)) for w in 
                strip_punctuation(text).split()])
        self.assertListEqual(newtexts, 
            [u"dit is tekst 1 die slechts kort is met hier u een accent en daar een gek teken".split(),
             u"dit is tekst 2 ook hier gekke tekens leestekens en woorden".split(),
             u"dit is de laatste tekst met wederom een aantal tekens niet om te schelden hoor en unicode tekens zoals au en".split()])

    def test_preprocess_min_word_len(self):
        self.assertListEqual(list(preprocess(self.texts, min_df=1, min_word_len=3)), 
            [u"dit tekst die slechts kort met hier een accent daar een gek teken".split(),
             u"dit tekst ook hier gekke tekens leestekens woorden".split(),
             u"dit laatste tekst met wederom een aantal tekens niet schelden hoor unicode tekens zoals".split()])

    def test_preprocess_min_df(self):
        self.assertListEqual(list(preprocess(self.texts, min_df=2)), 
            [u"dit tekst met hier een een".split(),
             u"dit tekst hier tekens".split(),
             u"dit tekst met een tekens tekens".split()])

    def test_preprocess_max_df(self):
        self.assertListEqual(list(preprocess(self.texts, min_df=1, max_df=0.9)),
            [u"die slechts kort met hier een accent daar een gek teken".split(),
             u"ook hier gekke tekens leestekens woorden".split(),
             u"laatste met wederom een aantal tekens niet schelden hoor unicode tekens zoals".split()])

    def test_preprocess_min_settings(self):
        self.assertListEqual(list(preprocess(self.texts, strip_accents='unicode', min_df=1, max_df=1.0, min_word_len=1)),
            [u"dit is tekst 1 die slechts kort is met hier u een accent en daar een gek teken".split(),
             u"dit is tekst 2 ook hier gekke tekens leestekens en woorden".split(),
             u"dit is de laatste tekst met wederom een aantal tekens niet om te schelden hoor en unicode tekens zoals au en".split()])

    def test_readline(self):
        self.assertListEqual(cleanfile(self.file, lambda f: preprocess(f), label_df=2),
            [(u'ID1', u'L1,L2', u"dit tekst met hier een een".split()),
             (u'ID2', u'L2', u"dit tekst hier tekens".split()),
             (u'ID3', u'L1', u"dit tekst met een tekens tekens".split())])


if __name__ == '__main__':
    unittest.main()