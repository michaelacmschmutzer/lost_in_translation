#!/usr/bin/env python3

import unittest
import numpy as np
import evolution as evo
from scipy.stats import multinomial

"""Unit tests for evolution.py"""

# preparations (imaginary amino acids, codons, mistranslation rates)
evo.amino_acids = ["A", "B", "C"]
evo.codontable = {"A": ["GCA","GCC","GCT","GCG"],
                  "B": ["TTT","TTA","TTC","TTG"],
                  "C": ["TGT","TGC"]}
evo.transltable = {"AGA": "A",
                   "AGC": "A",
                   "GCA": "A",
                   "GCC": "A",
                   "GCT": "A",
                   "GCG": "A",
                   "GGA": "A",
                   "GGC": "A",
                   "CGA": "B",
                   "CGC": "B",
                   "TTT": "B",
                   "TTA": "B",
                   "TTC": "B",
                   "TTG": "B",
                   "TAC": "C",
                   "TCC": "C",
                   "TGT": "C",
                   "TGG": "C",
                   "TCT": "C",
                   "TCA": "C",
                   "TGC": "C",
                   "TAA": "*",
                   "TGA": "*",
                   "TAG": "*"
                   }
evo.misrates = {"GCA": {"B": [0.20,1], "C": [0.05,1]},
                "GCC": {"B": [0.20,1], "C": [0.05,1]},
                "GCT": {"B": [0.20,1], "C": [0.05,1]},
                "GCG": {"B": [0.20,1], "C": [0.05,1]},
                "TTT": {"A": [0.07,1], "C": [0.03,1]},
                "TTA": {"A": [0.07,1], "C": [0.03,1]},
                "TTC": {"A": [0.07,1], "C": [0.03,1]},
                "TTG": {"A": [0.07,1], "C": [0.03,1]},
                "TGT": {"A": [0.08,1], "B": [0.12,1]},
                "TGC": {"A": [0.08,1], "B": [0.12,1]},
                }
fitdict = {"AA": 1,
           "AB": 0.9,
           "BA": 0.95,
           "AC": 0.8,
           "CA": 0.91,
           "CB": 0.12,
           "BC": 0.1,
           "BB": 0.2,
           "CC": 1.2
           }
# Use a wild type that translates to "AA"
wtseq = "GCAGCT"
# And a mutant that translates to "AC"
mtseq = "GCATCT"

class TestEvo(unittest.TestCase):

    def test_expected_mistranslation(self):
        p,f = evo.expected_mistranslation(wtseq, fitdict)
        exp_p = np.array([0.5625, 0.15, 0.0375, 0.15, 0.04, 0.01, 0.0375, 0.01, 0.0025])
        exp_f = np.array([1.    , 0.9 , 0.8   , 0.95, 0.2 , 0.1 , 0.91  , 0.12, 1.2   ])
        for i in range(len(p)):
            self.assertAlmostEqual(exp_p[i], p[i], places=6)
            self.assertAlmostEqual(exp_f[i], f[i], places=6)

    def test_prune_probabilities(self):
        p,f = evo.prune_probabilities(np.array([0.001, 0.999]), np.array([1,0.5]), threshold=0.01)
        self.assertAlmostEqual([0.999],p, places=6)
        self.assertAlmostEqual([0.5],f, places=6)

    def test_expected_fitness_moments(self):
        exp_p = np.array([0.5625, 0.15, 0.0375, 0.15, 0.04, 0.01, 0.0375, 0.01, 0.0025])
        exp_f = np.array([1.    , 0.9 , 0.8   , 0.95, 0.2 , 0.1 , 0.91  , 0.12, 1.2   ])
        expfm = 0.917325
        expfv = 0.003839
        fmean, fvar = evo.expected_fitness_moments(10, exp_p, exp_f)
        self.assertAlmostEqual(expfm, fmean, places=6)
        self.assertAlmostEqual(expfv, fvar, places=6)

    def test_stochastic_mistranslation(self):
        np.random.seed(0)
        exp_f = np.array([0.99, 0.88, 0.865])
        f = evo.stochastic_mistranslation(3, 10, wtseq, fitdict)
        for i in range(len(f)):
            self.assertAlmostEqual(exp_f[i], f[i], places=6)

    def test_reverse_translate(self):
        pass

    def test_find_neighbours(self):
        exp = ['AGA', 'GGA', 'CGA', 'TTA', 'TCA', 'TGT', 'TGG', 'TGC']
        neigh = evo.find_neighbours("TGA")
        self.assertTrue(all([sq in exp for sq in neigh]))

if __name__ == "__main__":
    unittest.main()
