# Notebook Migration Notes

The original public repository contained:

- `Exchange.ipynb`
- `ErOEr1+.csv`

Those files are intentionally preserved.  The importable package in
`src/superexchange` was extracted from the notebook and lightly wrapped so it can
be installed, smoke-tested, and called from scripts.

The original method names are preserved for compatibility with the notebook.
The full symbolic workflow still requires Wolfram Client for Python and a local
Wolfram kernel.  Smoke tests cover data loading, geometry helpers, quantum-number
parsing, tensor utilities, and clean optional-dependency behavior.
