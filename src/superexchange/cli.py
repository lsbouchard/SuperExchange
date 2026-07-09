"""Command line interface for SuperExchange."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import example_data_path
from .exchange import Exchange


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="superexchange",
        description="Inspect or run a SuperExchange molecule data sheet.",
    )
    parser.add_argument(
        "--data",
        default=str(example_data_path()),
        help="Path to a molecule CSV/XLSX file. Defaults to the bundled ErOEr1+ example.",
    )
    parser.add_argument(
        "--wolfram-path",
        default="",
        help="Optional path to a Wolfram kernel executable.",
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="Run the full symbolic superexchange calculation. Requires Wolfram.",
    )
    parser.add_argument("--atom-a", default="Er1", help="First endpoint atom.")
    parser.add_argument("--atom-b", default="Er2", help="Second endpoint atom.")
    parser.add_argument(
        "--set-atom",
        action="append",
        default=[],
        metavar="ATOM:X,Y,Z",
        help="Override an atom coordinate before running, for example O1:3.65,0,3.65.",
    )
    return parser


def _parse_atom_update(value: str) -> tuple[str, tuple[float, float, float]]:
    try:
        atom, xyz_text = value.split(":", 1)
        xyz = tuple(float(part.strip()) for part in xyz_text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "atom updates must look like ATOM:X,Y,Z"
        ) from exc
    if len(xyz) != 3:
        raise argparse.ArgumentTypeError("atom updates must include three coordinates")
    return atom.strip(), xyz  # type: ignore[return-value]


def _print_inspection(exchange: Exchange, data_path: Path) -> None:
    print(f"Data file: {data_path}")
    print("Atoms:")
    for _, row in exchange.molecule.iterrows():
        print(
            f"  {row['Atom']:>4s}  class={row['Class']!s:<6s} "
            f"xyz=({float(row['X']):.6g}, {float(row['Y']):.6g}, {float(row['Z']):.6g}) "
            f"bonds={row['Bondings']}"
        )
    print("Endpoint atoms:", ", ".join(exchange.get_Start_End_Atoms()))
    angles = exchange.get_All_Bond_Angles("deg")
    if not angles.empty:
        print("Bond angles:")
        for _, row in angles.iterrows():
            print(f"  {row['Bond']}: {float(row['Angle(degrees)']):.6g} deg")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_path = Path(args.data).expanduser()

    exchange = Exchange(str(data_path), wolfram_path=args.wolfram_path)
    for update in args.set_atom:
        atom, xyz = _parse_atom_update(update)
        exchange.Set_Atom_XYZ(atom, xyz)

    _print_inspection(exchange, data_path)

    if args.run_full:
        exchange.Superexchange_Terms()
        matrix = exchange.get_Total_Exchange_Matrix(args.atom_a, args.atom_b)
        print("Total exchange matrix:")
        print(np.array2string(matrix, precision=8, suppress_small=False))
    return 0
