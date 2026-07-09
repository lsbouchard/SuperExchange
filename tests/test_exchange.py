from __future__ import annotations

import math
import unittest

import numpy as np

from superexchange import Exchange, example_data_path, pylist_to_mathematica
from superexchange.exchange import WolframLanguageSession


class ExchangeSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exchange = Exchange(str(example_data_path()))

    def test_loads_bundled_data(self) -> None:
        self.assertEqual(list(self.exchange.molecule["Atom"]), ["Er1", "Er2", "O1"])
        self.assertEqual(self.exchange.get_Start_End_Atoms(), ["Er1", "Er2"])
        self.assertEqual(self.exchange.get_Bonds("O1"), ["Er1", "Er2"])

    def test_geometry_helpers(self) -> None:
        self.exchange.Set_Atom_XYZ("O1", (3.65, 0.0, 3.65))
        self.assertEqual(self.exchange.get_Displacement("Er1", "Er2"), [7.3, 0, 0])
        angle = self.exchange.get_Bond_Angle(["Er1", "O1", "Er2"], "deg")
        self.assertTrue(math.isclose(angle, 90.0, rel_tol=1e-12, abs_tol=1e-12))
        angles = self.exchange.get_All_Bond_Angles("deg")
        self.assertEqual(angles.to_dict("records"), [{"Bond": "Er1-O1-Er2", "Angle(degrees)": 90.0}])

    def test_quantum_number_parsing(self) -> None:
        self.assertEqual(self.exchange.get_Quantum_Numbers("2p"), [2, 1, [-1, 0, 1]])
        self.assertEqual(self.exchange.get_Quantum_Numbers("4f-2"), [4, 3, -2])
        self.assertEqual(self.exchange.get_Orbital_Label(4, 3, -2), "4f-2")

    def test_tensor_utilities(self) -> None:
        matrix = np.eye(3)
        self.assertTrue(math.isclose(self.exchange.Frob_Norm(matrix), math.sqrt(3.0)))
        dip = self.exchange.Dip_Dip_Int("Er1", "Er2")
        self.assertEqual(dip.shape, (3, 3))
        self.assertTrue(np.allclose(dip, dip.T))

    def test_mathematica_list_export(self) -> None:
        text = pylist_to_mathematica([[1, 2, 3, 0.001]])
        self.assertEqual(text, "{{1,2,3,1.00*^-03}}")

    def test_wolfram_dependency_error_is_clear_when_missing(self) -> None:
        if WolframLanguageSession is not None:
            self.skipTest("Wolfram Client is installed in this environment")
        with self.assertRaisesRegex(RuntimeError, "wolframclient"):
            self.exchange.Init_Mathematica()


if __name__ == "__main__":
    unittest.main()
