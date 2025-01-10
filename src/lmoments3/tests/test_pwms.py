"""Test cases for lmoments3.distr module."""

import unittest

from numpy.testing import assert_almost_equal

import lmoments3 as lm
from lmoments3 import distr, stats  # noqa: F401


class PWMDistributionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.testdata = [89.8, 109.1, 202.2, 146.3, 212.3, 116.7, 109.1, 80.7, 127.4, 
                        138.8, 283.5, 85.6, 105.5, 118.0, 387.8, 80.7, 165.7, 111.6, 
                        134.4, 131.5, 102.0, 104.3, 242.5, 214.8, 144.6, 114.2, 98.3, 
                        102.8, 104.3, 196.2, 143.7]
        # cls.lmu = lm.lmom_ratios(cls.testdata)
        if cls.dist:
            cls.distr_f = getattr(distr, cls.dist)
        super().setUpClass()

    def assertAlmostEqual(self, first, second, places=6):  # noqa: N802
        return assert_almost_equal(first, second, decimal=places)
    
    def test_n_paras(self):
        if self.distr_f:
            n = self.distr_f.numargs + 2
            self.assertEqual(len(self.paras), n)

    def test_fit(self):
        if self.distr_f:
            result = self.distr_f.lmom_fit(self.testdata)
            for para in iter(self.paras):
                self.assertAlmostEqual(result[para], self.paras[para])


class TestFisk(PWMDistributionTestCase):
    dist = "fisk"
    paras = {"c": 2.5868007, "loc": 58.1812847, "scale": 67.2304251}
    correct_fit = [2.5868007, 58.1812847, 67.2304251]

    def test_sampwm(self):
        """
        References:
            M.I. Ahmad, C.D. Sinclair, A. Werritty, Log-logistic flood frequency analysis.
            (https://doi.org/10.1016/0022-1694(88)90015-7)
        """
        d = self.distr_f
        pwm = d._sampwm(self.testdata, nr_order=2)
        self.assertAlmostEqual(pwm, [145.3032258, 55.8119043, 33.7646530], places=6)
