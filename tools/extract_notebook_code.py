"""Extract the original Exchange class from Exchange.ipynb.

This helper is intentionally small and deterministic.  It preserves the
student notebook implementation while adding the package imports and a few
import-time safety fixes needed for public release.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Exchange.ipynb"
TARGET = ROOT / "src" / "superexchange" / "exchange.py"


HEADER = '''"""Core superexchange calculation routines.

This module packages the original implementation from ``Exchange.ipynb`` so it
can be imported, tested, and reused from scripts.  The method names intentionally
remain compatible with the notebook.

Full superexchange calculations require Wolfram Client for Python and a local
Wolfram kernel.  Geometry, parsing, and tensor utility methods are available
without Wolfram.
"""

from __future__ import annotations

import itertools as it
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from wolframclient.evaluation import WolframLanguageSession, parallel_evaluate
    from wolframclient.language import wl, wlexpr
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    WolframLanguageSession = None
    parallel_evaluate = None
    wl = None
    wlexpr = None


def _require_wolfram() -> None:
    """Raise a clear error when the optional Wolfram dependency is unavailable."""
    if WolframLanguageSession is None:
        raise RuntimeError(
            "Full superexchange calculations require the optional "
            "'wolframclient' package and access to a local Wolfram kernel. "
            "Install with: pip install 'superexchange[wolfram]'"
        )


'''


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][0]["source"])
    body = source[source.index("#Exchange Class") :]
    body = body.replace(
        "    def Init_Mathematica(self):\n"
        "        #If you need to input a wolfram path, it's done in the original call.\n"
        "        if self.wolfram_path == \"N/A\":\n"
        "            self.session = WolframLanguageSession()\n"
        "        else:\n"
        "            self.session = WolframLanguageSession(self.wolfram_path)\n",
        "    def Init_Mathematica(self):\n"
        "        _require_wolfram()\n"
        "        # If a Wolfram path is supplied, pass it through to the client.\n"
        "        if self.wolfram_path in (None, \"\", \"N/A\"):\n"
        "            self.session = WolframLanguageSession()\n"
        "        else:\n"
        "            self.session = WolframLanguageSession(self.wolfram_path)\n",
    )
    body = body.replace(
        "    def get_Molecule_Data(self, file_name):\n"
        "        #Just keeps track of current data sheet name and resets structured_pairs since the structure may\n"
        "        #have changed when differing sheets.\n"
        "        if file_name != self.data_sheet_name:\n"
        "            self.data_sheet_name = file_name\n"
        "            self.structured_pairs = []\n"
        "        if \".xlsx\" in file_name:\n"
        "            self.molecule = pd.read_excel(file_name)\n"
        "        if \".csv\" in file_name:\n"
        "            self.molecule = pd.read_csv(file_name)\n"
        "        else:\n"
        "            return 'Error: Please use File Type .csv or .xlsx'\n",
        "    def get_Molecule_Data(self, file_name):\n"
        "        # Keep track of the active data sheet and reset structured pairs if it changes.\n"
        "        data_path = Path(file_name).expanduser()\n"
        "        if file_name != self.data_sheet_name:\n"
        "            self.data_sheet_name = file_name\n"
        "            self.structured_pairs = []\n"
        "        suffix = data_path.suffix.lower()\n"
        "        if suffix == \".xlsx\":\n"
        "            self.molecule = pd.read_excel(data_path)\n"
        "        elif suffix == \".csv\":\n"
        "            self.molecule = pd.read_csv(data_path)\n"
        "        else:\n"
        "            raise ValueError(\"Please use file type .csv or .xlsx\")\n",
    )
    body = body.replace("return session.evaluate(", "return self.session.evaluate(")
    body = body.replace(
        "        #Used to load up all used mathematica functions.\n"
        "        self.session = 0\n"
        "        self.Init_Mathematica()\n",
        "        # Wolfram is initialized lazily so data inspection works without Mathematica.\n"
        "        self.session = None\n",
    )
    body = body.replace(
        "    def Init_Mathematica(self):\n"
        "        _require_wolfram()\n",
        "    def Init_Mathematica(self):\n"
        "        if self.session is not None:\n"
        "            return\n"
        "        _require_wolfram()\n",
    )
    body = body.replace(
        "    def get_Coords(self, atom):\n"
        "        return [float(self.molecule[\"X\"][self.Atom_Index(atom)]), self.molecule[\"Y\"][self.Atom_Index(atom)], \n"
        "                self.molecule[\"Z\"][self.Atom_Index(atom)]]\n",
        "    def get_Coords(self, atom):\n"
        "        return [\n"
        "            float(self.molecule[\"X\"][self.Atom_Index(atom)]),\n"
        "            float(self.molecule[\"Y\"][self.Atom_Index(atom)]),\n"
        "            float(self.molecule[\"Z\"][self.Atom_Index(atom)]),\n"
        "        ]\n",
    )
    body = body.replace("q_num[1]", "q_nums[1]").replace("q_num[0]", "q_nums[0]")
    body = body.replace(
        "                        bond_angle_list[f\"Angle({unit})\"] += [get_Bond_Angles(molecule, \n"
        "                                                                     [bonding_list[i], atom, bonding_list[j]], deg_rad)]\n",
        "                        bond_angle_list[f\"Angle({unit})\"] += [\n"
        "                            self.get_Bond_Angle([bonding_list[i], atom, bonding_list[j]], deg_rad)\n"
        "                        ]\n",
    )
    body = body.replace(
        "            self.charge_transfer_energies[f\"{atom_from_class}-{atom_to_class}\"] = self.session.evaluate",
        "            self.Init_Mathematica()\n"
        "            self.charge_transfer_energies[f\"{atom_from_class}-{atom_to_class}\"] = self.session.evaluate",
    )
    body = body.replace(
        "        else:\n"
        "            #list_term = list(self.session.evaluate(f\"AngularVector[{n},{l},{m_bra}, {n},{l},{m_ket}]\"))",
        "        else:\n"
        "            self.Init_Mathematica()\n"
        "            #list_term = list(self.session.evaluate(f\"AngularVector[{n},{l},{m_bra}, {n},{l},{m_ket}]\"))",
    )
    body = body.replace(
        "    def Calculate_All_t(self):\n"
        "        #Gets all the pairs",
        "    def Calculate_All_t(self):\n"
        "        self.Init_Mathematica()\n"
        "        #Gets all the pairs",
    )
    body = body.replace(
        "    def Calculate_All_de(self):\n"
        "        exchange_atoms",
        "    def Calculate_All_de(self):\n"
        "        self.Init_Mathematica()\n"
        "        exchange_atoms",
    )
    body = body.replace(
        "    def Calculate_All_Neigbor_de(self):\n"
        "        #Initialize",
        "    def Calculate_All_Neigbor_de(self):\n"
        "        self.Init_Mathematica()\n"
        "        #Initialize",
    )
    body = body.replace(
        "        self.session.terminate()\n"
        "        self.Init_Mathematica()\n",
        "        self.session.terminate()\n"
        "        self.session = None\n"
        "        self.Init_Mathematica()\n",
    )
    body = body.replace(
        "        print(f\"Exchange Terms Complete: {round((time.perf_counter() - start)/60, 2)} min\")\n"
        "        self.session.terminate()\n",
        "        print(f\"Exchange Terms Complete: {round((time.perf_counter() - start)/60, 2)} min\")\n"
        "        self.session.terminate()\n"
        "        self.session = None\n",
    )
    body = body.replace("#Exchange Class\n", "")
    TARGET.write_text(HEADER + body)


if __name__ == "__main__":
    main()
