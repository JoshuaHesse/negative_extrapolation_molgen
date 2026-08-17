from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import posebusters as pb
import semlaflow.scriptutil as semla_util
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from semlaflow import predict as semla_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate molecules with a pretrained SemlaFlow checkpoint and summarize "
            "RDKit/PoseBusters validity. This is a baseline probe before implementing "
            "NEON fine-tuning for a 3D generator."
        )
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--data-path", required=True, help="Path to SemlaFlow processed smol folder.")
    parser.add_argument("--dataset", default="geom-drugs", choices=["geom-drugs", "qm9"])
    parser.add_argument("--dataset-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="results/external/semlaflow/geom_drugs_smoke")
    parser.add_argument("--save-prefix", default="semlaflow_geom_drugs")
    parser.add_argument("--n-mols", type=int, default=100)
    parser.add_argument("--batch-cost", type=int, default=4096)
    parser.add_argument("--integration-steps", type=int, default=100)
    parser.add_argument("--cat-sampling-noise-level", type=int, default=1)
    parser.add_argument("--ode-sampling-strategy", default="log")
    parser.add_argument("--bucket-cost-scale", default="linear")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--posebusters-config", default="mol")
    parser.add_argument("--posebusters-workers", type=int, default=4)
    parser.add_argument("--skip-posebusters", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


def sanitize_copy(mol: Chem.Mol) -> Chem.Mol | None:
    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
        return copy
    except Exception:
        return None


def canonical_smiles(mol: Chem.Mol) -> str | None:
    sanitized = sanitize_copy(mol)
    if sanitized is None:
        return None
    try:
        return Chem.MolToSmiles(sanitized, canonical=True)
    except Exception:
        return None


def scaffold_smiles(mol: Chem.Mol) -> str | None:
    sanitized = sanitize_copy(mol)
    if sanitized is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(sanitized)
        if scaffold.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def rdkit_row(index: int, mol: Chem.Mol | None) -> dict:
    if mol is None:
        return {
            "index": index,
            "rdkit_mol_present": False,
            "rdkit_sanitizes": False,
            "canonical_smiles": None,
            "scaffold_smiles": None,
        }

    sanitized = sanitize_copy(mol)
    row = {
        "index": index,
        "rdkit_mol_present": True,
        "rdkit_sanitizes": sanitized is not None,
        "canonical_smiles": canonical_smiles(mol),
        "scaffold_smiles": scaffold_smiles(mol),
        "n_atoms": mol.GetNumAtoms(),
    }
    if sanitized is not None:
        row.update(
            {
                "qed": float(QED.qed(sanitized)),
                "mw": float(Descriptors.MolWt(sanitized)),
                "logp": float(Crippen.MolLogP(sanitized)),
                "tpsa": float(rdMolDescriptors.CalcTPSA(sanitized)),
                "rings": int(rdMolDescriptors.CalcNumRings(sanitized)),
                "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(sanitized)),
                "fsp3": float(rdMolDescriptors.CalcFractionCSP3(sanitized)),
            }
        )
    return row


def sample_semlaflow(args: argparse.Namespace) -> list[Chem.Mol | None]:
    semla_util.disable_lib_stdout()
    semla_util.configure_fs()

    vocab = semla_util.build_vocab()
    model_args = argparse.Namespace(
        ckpt_path=args.ckpt_path,
        data_path=args.data_path,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        n_molecules=args.n_mols,
        batch_cost=args.batch_cost,
        integration_steps=args.integration_steps,
        cat_sampling_noise_level=args.cat_sampling_noise_level,
        ode_sampling_strategy=args.ode_sampling_strategy,
        bucket_cost_scale=args.bucket_cost_scale,
    )
    dm = semla_predict.dm_from_ckpt(model_args, vocab)
    model = semla_predict.load_model(model_args, vocab)
    molecules, _ = semla_util.generate_molecules(
        model,
        dm,
        args.integration_steps,
        args.ode_sampling_strategy,
    )
    return molecules


def write_sdf(mols: list[Chem.Mol | None], path: Path) -> int:
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)
    n_written = 0
    for mol in mols:
        if mol is None:
            continue
        writer.write(mol)
        n_written += 1
    writer.close()
    return n_written


def run_posebusters(mols: list[Chem.Mol | None], args: argparse.Namespace) -> pd.DataFrame:
    present = [mol for mol in mols if mol is not None]
    if not present:
        return pd.DataFrame()
    buster = pb.PoseBusters(config=args.posebusters_config, max_workers=args.posebusters_workers)
    return buster.bust(present, None, None).reset_index(drop=True)


def bool_fraction(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.fillna(False).astype(bool).mean())


def summarize(rdkit_df: pd.DataFrame, posebusters_df: pd.DataFrame, args: argparse.Namespace, n_sdf_written: int) -> dict:
    valid_smiles = rdkit_df.loc[rdkit_df["canonical_smiles"].notna(), "canonical_smiles"]
    unique_smiles = valid_smiles.drop_duplicates()
    summary = {
        "model_name": "semlaflow",
        "checkpoint": Path(args.ckpt_path).name,
        "dataset": args.dataset,
        "seed": args.seed,
        "n_sampled": args.n_mols,
        "n_rdkit_mol_present": int(rdkit_df["rdkit_mol_present"].sum()),
        "n_sdf_written": int(n_sdf_written),
        "rdkit_mol_present_fraction": float(rdkit_df["rdkit_mol_present"].mean()),
        "rdkit_sanitizes_fraction": float(rdkit_df["rdkit_sanitizes"].mean()),
        "n_valid_canonical_smiles": int(valid_smiles.shape[0]),
        "valid_canonical_smiles_fraction": float(valid_smiles.shape[0] / args.n_mols),
        "unique_valid_fraction": float(unique_smiles.shape[0] / valid_smiles.shape[0]) if valid_smiles.shape[0] else float("nan"),
        "unique_valid_yield": float(unique_smiles.shape[0] / args.n_mols),
        "unique_scaffold_count": int(rdkit_df.loc[rdkit_df["scaffold_smiles"].notna(), "scaffold_smiles"].nunique()),
    }
    for column in ["qed", "mw", "logp", "tpsa", "rings", "aromatic_rings", "fsp3"]:
        values = pd.to_numeric(rdkit_df.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        summary[f"{column}_mean"] = float(values.mean()) if not values.empty else float("nan")
        summary[f"{column}_median"] = float(values.median()) if not values.empty else float("nan")
    if not posebusters_df.empty:
        for column in posebusters_df.columns:
            summary[f"posebusters_{column}_fraction"] = bool_fraction(posebusters_df[column])
        summary["posebusters_all_checks_fraction"] = float(
            posebusters_df.fillna(False).astype(bool).all(axis=1).mean()
        )
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mols = sample_semlaflow(args)
    sdf_path = output_dir / f"{args.save_prefix}_seed_{args.seed}_{args.n_mols}.sdf"
    n_sdf_written = write_sdf(mols, sdf_path)

    rdkit_df = pd.DataFrame([rdkit_row(index, mol) for index, mol in enumerate(mols)])
    rdkit_path = output_dir / "rdkit_profile.csv"
    rdkit_df.to_csv(rdkit_path, index=False)

    posebusters_df = pd.DataFrame() if args.skip_posebusters else run_posebusters(mols, args)
    posebusters_path = output_dir / "posebusters_profile.csv"
    posebusters_df.to_csv(posebusters_path, index=False)

    summary = summarize(rdkit_df, posebusters_df, args, n_sdf_written)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    summary_table_path = output_dir / "summary.csv"
    pd.DataFrame([summary]).to_csv(summary_table_path, index=False)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {sdf_path}")
    print(f"Wrote {rdkit_path}")
    print(f"Wrote {posebusters_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
