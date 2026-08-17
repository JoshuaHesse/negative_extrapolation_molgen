#!/usr/bin/env python3
"""Export the code-defined liability SMARTS as an SI table."""

from __future__ import annotations

import argparse
from pathlib import Path

from neon_molgen.scoring import LIABILITY_SMARTS, PAPER_LIABILITY_FAMILIES

FAMILY_LABELS = {
    "reactive": "Reactive motifs",
    "chelator": "Metal-binding motifs",
    "charged_motif": "Charged motifs",
    "assay_interference": "Assay-interference motifs",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/publication/tables"),
    )
    return parser.parse_args()


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_smarts_table(output_path: Path) -> None:
    lines = [
        r"\begin{longtable}{p{0.20\linewidth}p{0.25\linewidth}p{0.45\linewidth}}",
        r"\caption{Exact study-defined SMARTS patterns used for the four confirmatory structural objectives. A family flag is positive if any listed pattern matches. Pattern names reproduce the identifiers in \texttt{neon\_molgen/scoring.py}.}\label{tab:liability-smarts}\\",
        r"\toprule",
        r"Family & Pattern & SMARTS \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{c}{\tablename\ \thetable\ -- continued}\\",
        r"\toprule",
        r"Family & Pattern & SMARTS \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        "\\multicolumn{3}{r}{Continued on next page}\\\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for family in PAPER_LIABILITY_FAMILIES:
        patterns = LIABILITY_SMARTS[family]
        for index, (name, smarts) in enumerate(patterns.items()):
            family_label = FAMILY_LABELS[family] if index == 0 else ""
            lines.append(
                f"{family_label} & {latex_escape(name.replace('_', ' '))} & "
                rf"\texttt{{\detokenize{{{smarts}}}}} \\"
            )
        lines.append(r"\addlinespace")
    lines.append(r"\end{longtable}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_smarts_table(args.output_dir / "si_liability_smarts.tex")
    print(f"Wrote SI SMARTS table to {args.output_dir}")


if __name__ == "__main__":
    main()
