"""Minimal SuperExchange example.

This script only inspects the bundled geometry.  The full symbolic exchange
calculation requires Wolfram Client for Python and a local Wolfram kernel.
"""

from __future__ import annotations

from superexchange import Exchange, example_data_path


def main() -> None:
    molecule = Exchange(str(example_data_path()))
    molecule.Set_Atom_XYZ("O1", (3.65, 0.0, 3.65))
    print(molecule.molecule[["Atom", "X", "Y", "Z", "Bondings", "Class"]])
    print("Er1-Er2 displacement:", molecule.get_Displacement("Er1", "Er2"))
    print("Er1-O1-Er2 angle:", molecule.get_Bond_Angle(["Er1", "O1", "Er2"], "deg"))


if __name__ == "__main__":
    main()
