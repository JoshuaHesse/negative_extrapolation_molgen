from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neon_molgen.scoring import score_smiles, summarize_scores

SMILES_COLUMNS = ("smiles", "SMILES", "canonical_smiles", "sampled_smiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score SMILES sampled from an external REINVENT model with local liability metrics."
    )
    parser.add_argument("--input", required=True, help="Input .smi/.smiles/.txt/.csv/.tsv file.")
    parser.add_argument("--output", required=True, help="Output scored CSV path.")
    parser.add_argument("--summary-output", default=None, help="Optional one-row summary CSV path.")
    parser.add_argument("--smiles-column", default=None, help="SMILES column for CSV/TSV input.")
    parser.add_argument(
        "--liability-column",
        default="reactive_hit",
        help="Hit column whose absence defines liability-free molecules for usable_yield.",
    )
    parser.add_argument(
        "--n-sampled",
        type=int,
        default=None,
        help="Original sampling budget. Defaults to the number of rows read from input.",
    )
    return parser.parse_args()


def read_smiles_table(path: Path, smiles_column: str | None = None) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator)
        column = smiles_column
        if column is None:
            column = next((name for name in SMILES_COLUMNS if name in frame.columns), None)
        if column is None or column not in frame.columns:
            raise ValueError(
                f"Could not find a SMILES column in {path}. "
                f"Use --smiles-column. Available columns: {list(frame.columns)}"
            )
        return frame[column].dropna().astype(str).tolist()

    smiles: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        smiles.append(stripped.split()[0])
    return smiles


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    smiles = read_smiles_table(input_path, smiles_column=args.smiles_column)
    scores = score_smiles(smiles)
    scores.to_csv(output_path, index=False)

    summary = pd.DataFrame(
        [
            summarize_scores(
                scores,
                liability_free_column=args.liability_column,
                n_sampled=args.n_sampled,
            )
        ]
    )
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"Wrote scored samples to {output_path}")


if __name__ == "__main__":
    main()
