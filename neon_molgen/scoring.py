from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, FilterCatalog, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer

RDLogger.DisableLog("rdApp.error")

LIABILITY_SMARTS = {
    "reactive": {
        "acid_chloride": "C(=O)Cl",
        "sulfonyl_chloride": "S(=O)(=O)Cl",
        "isocyanate": "N=C=O",
        "isothiocyanate": "N=C=S",
        "aldehyde": "[CX3H1](=O)[#6]",
        "epoxide": "[OX2r3]1[#6r3][#6r3]1",
        "aziridine": "[NX3r3]1[#6r3][#6r3]1",
        "michael_acceptor": "[C,c]=[C,c][C,S](=O)",
        "alpha_beta_unsaturated_carbonyl": "[C,c]=[C,c]C(=O)[#6,#7,#8]",
        "alpha_beta_unsaturated_nitrile": "[C,c]=[C,c]C#N",
        "nitroalkene": "[C,c]=[C,c][N+](=O)[O-]",
        "vinyl_sulfone_sulfoxide": "[C,c]=[C,c]S(=O)",
        "haloacetamide": "[Cl,Br,I]CC(=O)N",
        "maleimide": "O=C1NC(=O)C=C1",
        "alkyl_halide": "[CX4;!$(C(F)(F)F)][Cl,Br,I]",
    },
    "unstable": {
        "azide": "N=[N+]=[N-]",
        "peroxide": "[OX2][OX2]",
        "diazo": "[C,N]=[N+]=[N-]",
        "acyl_peroxide": "C(=O)O[OH,OR0]",
        "hemiacetal": "[CX4]([OX2H])([OX2])[#6]",
    },
    "chelator": {
        "catechol": "c([OH])c([OH])",
        "hydroxamic_acid": "C(=O)N[OH]",
        "amidoxime": "C(=N[OH])N",
        "thiol": "[SH]",
        "dithiocarbamate": "N(C(=S)S)",
        "beta_dicarbonyl": "C(=O)C[C,N,O,S]C(=O)",
    },
    "charged_motif": {
        "quaternary_ammonium": "[NX4+]",
        "sulfonate": "S(=O)(=O)[O-]",
        "phosphonate": "P(=O)([O-])[O-]",
        "carboxylate": "C(=O)[O-]",
        "zwitterion_proxy": "[+].[-]",
    },
    "assay_interference": {
        "rhodanine": "O=C1NC(=S)SC1",
        "quinone": "O=C1C=CC(=O)C=C1",
        "azo": "[#6]-N=N-[#6]",
        "nitro": "[$([NX3](=O)=O),$([N+](=O)[O-])]",
        "thiourea": "NC(=S)N",
        "polyphenol": "c([OH])cc([OH])",
    },
}

LIABILITY_FAMILIES = tuple(LIABILITY_SMARTS)
PAPER_LIABILITY_FAMILIES = (
    "reactive",
    "chelator",
    "charged_motif",
    "assay_interference",
)
CATALOG_NAMES = ("pains", "brenk", "nih", "zinc")
BRENK_TOP5_FILTERS = {
    "oxygen_nitrogen_single_bond": "Oxygen-nitrogen_single_bond",
    "aniline": "aniline",
    "imine_1": "imine_1",
    "two_halo_pyridine": "2-halo_pyridine",
    "nitro_group": "nitro_group",
}
@lru_cache(maxsize=1)
def make_filter_catalogs() -> dict[str, FilterCatalog.FilterCatalog]:
    definitions = {
        "pains": [
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C,
        ],
        "brenk": [FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK],
        "nih": [FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH],
        "zinc": [FilterCatalog.FilterCatalogParams.FilterCatalogs.ZINC],
    }
    catalogs = {}
    for name, values in definitions.items():
        params = FilterCatalog.FilterCatalogParams()
        for value in values:
            params.AddCatalog(value)
        catalogs[name] = FilterCatalog.FilterCatalog(params)
    return catalogs


@lru_cache(maxsize=1)
def make_liability_patterns() -> dict[str, dict[str, Chem.Mol]]:
    patterns = {}
    for family, smarts_by_name in LIABILITY_SMARTS.items():
        patterns[family] = {}
        for name, smarts in smarts_by_name.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                patterns[family][name] = pattern
    return patterns


def canonicalize_smiles_set(smiles: list[str]) -> set[str]:
    canonical = set()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical.add(Chem.MolToSmiles(mol, canonical=True))
    return canonical


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True)


def scaffold_counts_from_smiles(smiles: list[str] | set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = murcko_scaffold_smiles(mol)
        if scaffold:
            counts[scaffold] += 1
    return counts


def flat_lipophilic_score(logp: float, aromatic_rings: int, fsp3: float) -> float:
    logp_component = max(0.0, (logp - 3.0) / 3.0)
    aromatic_component = max(0.0, (float(aromatic_rings) - 2.0) / 3.0)
    fsp3_component = max(0.0, (0.25 - fsp3) / 0.25)
    return float(logp_component + aromatic_component + fsp3_component)


def score_smiles(smiles: list[str]) -> pd.DataFrame:
    rows = []
    catalogs = make_filter_catalogs()
    liability_patterns = make_liability_patterns()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        valid = mol is not None
        if valid:
            try:
                qed = float(QED.qed(mol))
                logp = float(Crippen.MolLogP(mol))
                mw = float(Descriptors.MolWt(mol))
                tpsa = float(rdMolDescriptors.CalcTPSA(mol))
                canonical = Chem.MolToSmiles(mol, canonical=True)
                scaffold = murcko_scaffold_smiles(mol)
                sa_score = float(sascorer.calculateScore(mol))
                aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
                rings = int(rdMolDescriptors.CalcNumRings(mol))
                fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
                formal_charge = int(Chem.GetFormalCharge(mol))
                hbd = int(rdMolDescriptors.CalcNumHBD(mol))
                hba = int(rdMolDescriptors.CalcNumHBA(mol))
                catalog_hits = {
                    f"{name}_hit": float(catalog.HasMatch(mol))
                    for name, catalog in catalogs.items()
                }
                alert_hit = float(any(catalog_hits[f"{name}_hit"] for name in CATALOG_NAMES))
                brenk_matches = {
                    match.GetDescription() for match in catalogs["brenk"].GetMatches(mol)
                }
                brenk_top5_hits = {
                    f"brenk_top5_{slug}_hit": float(description in brenk_matches)
                    for slug, description in BRENK_TOP5_FILTERS.items()
                }
                brenk_top5_hit = float(any(brenk_top5_hits.values()))
                liability_hits = {}
                liability_pattern_hits = {}
                for family, patterns in liability_patterns.items():
                    family_hits = {}
                    for pattern_name, pattern in patterns.items():
                        hit = float(mol.HasSubstructMatch(pattern))
                        family_hits[pattern_name] = hit
                        liability_pattern_hits[f"{family}_{pattern_name}_hit"] = hit
                    liability_hits[f"{family}_hit"] = float(any(family_hits.values()))
                liability_hit = float(any(liability_hits.values()))
                four_liability_hit = float(
                    any(liability_hits[f"{family}_hit"] for family in PAPER_LIABILITY_FAMILIES)
                )
                flat_score = flat_lipophilic_score(logp, aromatic_rings, fsp3)
            except Exception:
                valid = False
        if not valid:
            qed = np.nan
            logp = np.nan
            mw = np.nan
            tpsa = np.nan
            canonical = ""
            scaffold = ""
            sa_score = np.nan
            aromatic_rings = np.nan
            rings = np.nan
            fsp3 = np.nan
            formal_charge = np.nan
            hbd = np.nan
            hba = np.nan
            catalog_hits = {f"{name}_hit": np.nan for name in CATALOG_NAMES}
            alert_hit = np.nan
            brenk_top5_hits = {
                f"brenk_top5_{slug}_hit": np.nan for slug in BRENK_TOP5_FILTERS
            }
            brenk_top5_hit = np.nan
            liability_hits = {f"{family}_hit": np.nan for family in LIABILITY_FAMILIES}
            liability_pattern_hits = {
                f"{family}_{pattern_name}_hit": np.nan
                for family, patterns in LIABILITY_SMARTS.items()
                for pattern_name in patterns
            }
            liability_hit = np.nan
            four_liability_hit = np.nan
            flat_score = np.nan
        rows.append(
            {
                "smiles": smi,
                "canonical_smiles": canonical,
                "murcko_scaffold": scaffold,
                "valid": valid,
                "qed": qed,
                "logp": logp,
                "mw": mw,
                "tpsa": tpsa,
                "sa_score": sa_score,
                "aromatic_rings": aromatic_rings,
                "rings": rings,
                "fsp3": fsp3,
                "formal_charge": formal_charge,
                "hbd": hbd,
                "hba": hba,
                **catalog_hits,
                "alert_hit": alert_hit,
                "brenk_top5_hit": brenk_top5_hit,
                **brenk_top5_hits,
                **liability_hits,
                **liability_pattern_hits,
                "liability_hit": liability_hit,
                "four_liability_hit": four_liability_hit,
                "flat_lipophilic_score": flat_score,
            }
        )
    frame = pd.DataFrame(rows)
    if "reactive_hit" in frame:
        # Covalent warhead is a clearer medicinal-chemistry label for the
        # electrophilic/reactive SMARTS family used by this objective.
        frame["covalent_warhead_hit"] = frame["reactive_hit"]
        for pattern_name in LIABILITY_SMARTS["reactive"]:
            frame[f"covalent_warhead_{pattern_name}_hit"] = frame[
                f"reactive_{pattern_name}_hit"
            ]
    frame["property_penalty"] = (
        frame["mw"].sub(350).abs().div(350).fillna(1.0)
        + frame["logp"].sub(2.5).abs().div(5).fillna(1.0)
        + frame["tpsa"].sub(75).abs().div(150).fillna(1.0)
    )
    frame["bad_score"] = (1.0 - frame["qed"].fillna(0.0)) + frame["property_penalty"]
    return frame


def add_reference_scaffold_scores(
    scores: pd.DataFrame,
    reference_scaffold_counts: dict[str, int] | Counter[str],
    *,
    common_scaffold_min_count: int = 25,
) -> pd.DataFrame:
    frame = scores.copy()
    if "murcko_scaffold" not in frame:
        frame["murcko_scaffold"] = ""
        for idx, row in frame[frame["valid"]].iterrows():
            smiles = row.get("canonical_smiles")
            mol = Chem.MolFromSmiles(str(smiles)) if isinstance(smiles, str) else None
            if mol is not None:
                frame.at[idx, "murcko_scaffold"] = murcko_scaffold_smiles(mol)

    total = float(sum(reference_scaffold_counts.values()))
    counts = frame["murcko_scaffold"].map(lambda scaffold: int(reference_scaffold_counts.get(scaffold, 0)))
    frame["reference_scaffold_count"] = counts.astype(float)
    frame["reference_scaffold_fraction"] = counts.astype(float).div(total) if total else np.nan
    frame["reference_scaffold_hit"] = counts.gt(0).astype(float)
    frame["common_reference_scaffold_hit"] = counts.ge(common_scaffold_min_count).astype(float)
    frame.loc[~frame["valid"], [
        "reference_scaffold_count",
        "reference_scaffold_fraction",
        "reference_scaffold_hit",
        "common_reference_scaffold_hit",
    ]] = np.nan
    return frame


def summarize_scores(
    scores: pd.DataFrame,
    *,
    reference_canonical_smiles: set[str] | None = None,
    liability_free_column: str | None = None,
    n_sampled: int | None = None,
) -> dict[str, float]:
    valid = scores[scores["valid"]]
    unique_smiles = set(valid["canonical_smiles"])
    valid_unique = valid.drop_duplicates("canonical_smiles").copy()
    if reference_canonical_smiles is not None:
        valid_unique_novel = valid_unique[
            ~valid_unique["canonical_smiles"].isin(reference_canonical_smiles)
        ].copy()
    else:
        # No novelty reference is available for some external pretrained models.
        # In that case usable_yield is a fixed-sampling-budget valid/unique/liability-free yield.
        valid_unique_novel = valid_unique.copy()

    if liability_free_column is None:
        liability_free_column = "liability_hit" if "liability_hit" in scores.columns else None

    n_sampled_value = int(n_sampled) if n_sampled is not None else int(len(scores))
    if liability_free_column is not None and liability_free_column in valid_unique_novel:
        liability_free = valid_unique_novel[valid_unique_novel[liability_free_column].eq(0)]
        n_valid_unique_novel_liability_free = int(len(liability_free))
        usable_yield = (
            float(n_valid_unique_novel_liability_free / n_sampled_value)
            if n_sampled_value
            else np.nan
        )
    else:
        n_valid_unique_novel_liability_free = np.nan
        usable_yield = np.nan

    def mean_or_nan(column: str) -> float:
        return float(valid[column].mean()) if column in valid and len(valid) else np.nan

    def p90_or_nan(column: str) -> float:
        return float(valid[column].quantile(0.90)) if column in valid and len(valid) else np.nan

    summary = {
        "n": float(len(scores)),
        "n_sampled": float(n_sampled_value),
        "n_valid": float(len(valid)),
        "n_valid_unique": float(len(valid_unique)),
        "n_valid_unique_novel": float(len(valid_unique_novel)),
        "n_valid_unique_novel_liability_free": float(n_valid_unique_novel_liability_free),
        # Fixed-sampling-budget yield of valid, unique, novel, liability-free molecules.
        "usable_yield": usable_yield,
        "usable_yield_liability_column": liability_free_column,
        "valid_fraction": float(scores["valid"].mean()) if len(scores) else np.nan,
        "unique_fraction": float(len(unique_smiles) / len(valid)) if len(valid) else np.nan,
        "qed_mean": float(valid["qed"].mean()) if len(valid) else np.nan,
        "qed_median": float(valid["qed"].median()) if len(valid) else np.nan,
        "qed_p10": float(valid["qed"].quantile(0.10)) if len(valid) else np.nan,
        "qed_p90": float(valid["qed"].quantile(0.90)) if len(valid) else np.nan,
        "qed_ge_0.8_fraction": float(valid["qed"].ge(0.8).mean()) if len(valid) else np.nan,
        "qed_ge_0.9_fraction": float(valid["qed"].ge(0.9).mean()) if len(valid) else np.nan,
        "bad_score_mean": float(valid["bad_score"].mean()) if len(valid) else np.nan,
        "mw_mean": float(valid["mw"].mean()) if len(valid) else np.nan,
        "logp_mean": float(valid["logp"].mean()) if len(valid) else np.nan,
        "tpsa_mean": float(valid["tpsa"].mean()) if len(valid) else np.nan,
        "logp_le_2_fraction": float(valid["logp"].le(2).mean()) if len(valid) else np.nan,
        "logp_gt_2_fraction": float(valid["logp"].gt(2).mean()) if len(valid) else np.nan,
        "sa_score_mean": mean_or_nan("sa_score"),
        "sa_score_p90": p90_or_nan("sa_score"),
        "aromatic_rings_mean": mean_or_nan("aromatic_rings"),
        "rings_mean": mean_or_nan("rings"),
        "fsp3_mean": mean_or_nan("fsp3"),
        "formal_charge_abs_mean": float(valid["formal_charge"].abs().mean())
        if "formal_charge" in valid and len(valid)
        else np.nan,
        "hbd_mean": mean_or_nan("hbd"),
        "hba_mean": mean_or_nan("hba"),
        "flat_lipophilic_score_mean": mean_or_nan("flat_lipophilic_score"),
        "alert_hit_fraction": mean_or_nan("alert_hit"),
        "liability_hit_fraction": mean_or_nan("liability_hit"),
        "four_liability_hit_fraction": mean_or_nan("four_liability_hit"),
        "brenk_top5_hit_fraction": mean_or_nan("brenk_top5_hit"),
        "covalent_warhead_hit_fraction": mean_or_nan("covalent_warhead_hit"),
        "reference_scaffold_count_mean": mean_or_nan("reference_scaffold_count"),
        "reference_scaffold_count_p90": p90_or_nan("reference_scaffold_count"),
        "reference_scaffold_hit_fraction": mean_or_nan("reference_scaffold_hit"),
        "common_reference_scaffold_hit_fraction": mean_or_nan("common_reference_scaffold_hit"),
        "novel_reference_scaffold_fraction": (
            1.0 - mean_or_nan("reference_scaffold_hit")
            if "reference_scaffold_hit" in valid and len(valid)
            else np.nan
        ),
    }
    for name in CATALOG_NAMES:
        summary[f"{name}_hit_fraction"] = mean_or_nan(f"{name}_hit")
    for slug in BRENK_TOP5_FILTERS:
        summary[f"brenk_top5_{slug}_hit_fraction"] = mean_or_nan(
            f"brenk_top5_{slug}_hit"
        )
    for family in LIABILITY_FAMILIES:
        summary[f"{family}_hit_fraction"] = mean_or_nan(f"{family}_hit")
        for pattern_name in LIABILITY_SMARTS[family]:
            summary[f"{family}_{pattern_name}_hit_fraction"] = mean_or_nan(
                f"{family}_{pattern_name}_hit"
            )
    for pattern_name in LIABILITY_SMARTS["reactive"]:
        summary[f"covalent_warhead_{pattern_name}_hit_fraction"] = mean_or_nan(
            f"covalent_warhead_{pattern_name}_hit"
        )
    for column in scores.columns:
        if not column.endswith("_hit"):
            continue
        fraction_column = f"{column}_fraction"
        if fraction_column not in summary:
            summary[fraction_column] = mean_or_nan(column)
    if reference_canonical_smiles is not None:
        novel_valid = valid[~valid["canonical_smiles"].isin(reference_canonical_smiles)]
        novel_unique = unique_smiles - reference_canonical_smiles
        summary["novel_fraction"] = float(len(novel_valid) / len(valid)) if len(valid) else np.nan
        summary["unique_novel_fraction"] = (
            float(len(novel_unique) / len(unique_smiles)) if unique_smiles else np.nan
        )
    return summary
