# SuperExchange

Research code for symmetry-guided superexchange tensor calculations in
rare-earth-doped silicon spin-qubit systems.

This repository began as a Jupyter notebook implementation and now includes an
installable Python package, a command line interface, example data, and smoke
tests.  The original notebook and CSV remain in the repository for provenance.

## Important Notice

This code is provided as-is for archival and research transparency.

There is no technical support.  The authors do not provide installation help,
debugging help, feature development, consulting, or guaranteed responses to
issues or pull requests.

Use this repository only if you are comfortable reading, validating, and
modifying research code independently.

## License

This repository is released under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

Commercial use is not permitted under this license.  For any commercial,
industrial, or product use, contact the copyright holder for a separate written
license.

## What Is Included

- `Exchange.ipynb`: original notebook implementation.
- `ErOEr1+.csv`: original example molecule data sheet.
- `src/superexchange`: importable Python package extracted from the notebook.
- `examples/basic_scan.py`: minimal package usage example.
- `tests`: smoke tests for loading, geometry helpers, tensor utilities, and
  optional-dependency behavior.
- `docs/notebook_migration.md`: notes on the notebook-to-package migration.

## Requirements

Basic package use:

- Python 3.10 or newer
- NumPy
- pandas

Full symbolic superexchange calculations additionally require:

- Wolfram Client for Python
- A local Wolfram kernel

The smoke tests do not require Wolfram.

## Installation

From a local clone:

```bash
python -m pip install .
```

For editable development:

```bash
python -m pip install -e .
```

To enable the Wolfram Python client dependency:

```bash
python -m pip install ".[wolfram]"
```

You must still have access to a local Wolfram kernel.

## Quick Start

Inspect the bundled example data:

```bash
python -m superexchange
```

Use the Python API:

```python
from superexchange import Exchange, example_data_path

molecule = Exchange(str(example_data_path()))
molecule.Set_Atom_XYZ("O1", (3.65, 0.0, 3.65))
print(molecule.get_Displacement("Er1", "Er2"))
print(molecule.get_Bond_Angle(["Er1", "O1", "Er2"], "deg"))
```

Run the example script:

```bash
PYTHONPATH=src python examples/basic_scan.py
```

## Full Calculation

The full superexchange workflow calls Wolfram through `wolframclient`.

```bash
python -m superexchange --run-full --data ErOEr1+.csv
```

If your Wolfram kernel is not on the default path:

```bash
python -m superexchange --run-full --wolfram-path /path/to/WolframKernel
```

Full calculations may be slow and are not covered by the lightweight smoke
tests.

## Smoke Tests

From the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

The tests intentionally avoid Wolfram so the repository can be checked on a
standard Python installation.

## Repository Status

This is research software, not a supported product.  Numerical assumptions,
units, scientific validity, and suitability for a particular project must be
checked by the user.
