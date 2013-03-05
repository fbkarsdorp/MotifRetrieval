import unittest

from TMI import *


class CompileIndexTest(unittest.TestCase):

    def test_split_index_entry(self):
        entries = ['B100.-B199. Magic animals', 'B90.Other mythical animals']
        self.assertEqual(map(split_index_entry, entries), [
            ('B100.-B199', 'Magic animals'), ('B90', 'Other mythical animals')])

    def test_rounddown(self):
        self.assertEqual(rounddown('A415'), 'A400')
        self.assertEqual(rounddown('A419'), 'A400')
        self.assertEqual(rounddown('A415', 10), 'A410')
        self.assertEqual(rounddown('A451', 10), 'A450')

    def test_find_best_parent(self):
        pass

    def test_spell_out_nodes(self):
        self.assertListEqual(spell_out_nodes('K411.1.3'), ['K411', 'K411.1'])
        self.assertListEqual(
            spell_out_nodes('Q456.1.4.2'), ['Q456', 'Q456.1', 'Q456.1.4'])

    def test_main_category_motif(self):
        self.assertTrue(main_category_motif(('A', 'DESCRIPTION')))
        self.assertFalse(main_category_motif('A1', 'DESCRIPTION'))

if __name__ == '__main__':
    unittest.main()
