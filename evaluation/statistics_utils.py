"""Lightweight peptide/PTM normalization helpers used by rerank scripts.

This module vendors the subset of the workspace-level ``statistics_utils.py``
that is required by the public MemNovo repository so scripts can run from a
fresh clone without depending on files outside the repo.
"""

from __future__ import annotations

import bisect
import re
from typing import Dict, List, Tuple


AMINO_ACID_MASSES = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C": 103.00919,
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
}

PTM_MASSES = {
    "acetyl": 42.010565,
    "carbamyl": 43.005814,
    "-nh3": -17.026549,
    "carbamyl)(-nh3": 25.980265,
    "ox": 15.99491,
    "deamide": 0.98401,
    "carbamidomethyl": 57.02146,
    "phos": 79.966331,
    "+42.011": 42.010565,
    "+42.010565": 42.010565,
    "+57.021": 57.02146,
    "+57.02146": 57.02146,
    "+0.984": 0.98401,
    "+0.98401": 0.98401,
    "+15.995": 15.99491,
    "+15.99491": 15.99491,
    "+79.966": 79.966331,
    "+79.966331": 79.966331,
    "-17.027": -17.026549,
    "-17.026549": -17.026549,
}


def normalize_peptide(peptide: str) -> str:
    """Strip PTM markup and normalize I/L."""
    if not peptide:
        return ""

    peptide = re.sub(r"^[+-]?\d+\.?\d*", "", peptide)
    peptide = re.sub(r"\[.*?\]", "", peptide)
    peptide = re.sub(r"\+[\d.]+", "", peptide)
    peptide = re.sub(r"-[\d.]+", "", peptide)
    peptide = re.sub(r"\(.*?\)", "", peptide)
    return peptide.replace("I", "L")


def convert_ultraprot_to_mass_format(peptide: str) -> str:
    if not peptide:
        return peptide

    peptide = re.sub(r"^\(acetyl\)", "+42.011", peptide)
    peptide = re.sub(r"C\(carbamidomethyl\)", "C+57.021", peptide)
    peptide = re.sub(r"Q\(deamide\)", "Q+0.984", peptide)
    peptide = re.sub(r"N\(deamide\)", "N+0.984", peptide)
    peptide = re.sub(r"M\(ox\)", "M+15.995", peptide)
    return peptide


def convert_casanovo_to_mass_format(peptide: str) -> str:
    if not peptide:
        return peptide

    peptide = re.sub(r"^\[Acetyl\]-", "+42.011", peptide)
    peptide = re.sub(r"^\[Carbamyl\]-", "+43.006", peptide)
    peptide = re.sub(r"<deam>", "+0.984", peptide)
    peptide = re.sub(r"<cmm>", "+57.021", peptide)
    peptide = re.sub(r"<ox>", "+15.995", peptide)
    peptide = re.sub(r"(\w)\[Deamidated\]", r"\1+0.984", peptide)
    peptide = re.sub(r"(\w)\[Carbamidomethyl\]", r"\1+57.021", peptide)
    peptide = re.sub(r"(\w)\[Oxidation\]", r"\1+15.995", peptide)
    peptide = re.sub(r"pyro-", "", peptide)
    return peptide


def convert_unimod_to_mass_format(peptide: str) -> str:
    if not peptide:
        return peptide

    unimod_map = {
        "UNIMOD:7": "+0.984",
        "UNIMOD:4": "+57.021",
        "UNIMOD:1": "+42.011",
        "UNIMOD:5": "+43.006",
        "UNIMOD:21": "+79.966",
        "UNIMOD:35": "+15.995",
        "UNIMOD:385": "-17.027",
    }

    peptide = re.sub(r"^\[UNIMOD:1\]", "+42.011", peptide)
    peptide = re.sub(r"^\[UNIMOD:5\]", "+43.006", peptide)
    peptide = re.sub(r"\[UNIMOD:385\]", "-17.027", peptide)

    for unimod, mass in unimod_map.items():
        if unimod in {"UNIMOD:1", "UNIMOD:5", "UNIMOD:385"}:
            continue
        peptide = re.sub(rf"(\w)\[{re.escape(unimod)}\]", rf"\1{mass}", peptide)
    return peptide


def convert_mass_offset_brackets_to_standard(peptide: str) -> str:
    if not peptide:
        return peptide

    mass_mappings = [
        (r"C\(\+57\.0?2\)", "C+57.021"),
        (r"C\(\+57\.021\)", "C+57.021"),
        (r"C\(\+57\.02146\)", "C+57.021"),
        (r"Q\(\+\.98\)", "Q+0.984"),
        (r"Q\(\+0\.98\)", "Q+0.984"),
        (r"Q\(\+0\.984\)", "Q+0.984"),
        (r"Q\(\+0\.98401\)", "Q+0.984"),
        (r"N\(\+\.98\)", "N+0.984"),
        (r"N\(\+0\.98\)", "N+0.984"),
        (r"N\(\+0\.984\)", "N+0.984"),
        (r"N\(\+0\.98401\)", "N+0.984"),
        (r"M\(\+15\.99\)", "M+15.995"),
        (r"M\(\+15\.995\)", "M+15.995"),
        (r"M\(\+15\.99491\)", "M+15.995"),
        (r"S\(\+79\.966\)", "S+79.966"),
        (r"T\(\+79\.966\)", "T+79.966"),
        (r"Y\(\+79\.966\)", "Y+79.966"),
        (r"^\(\+42\.011\)", "+42.011"),
        (r"^\(\+42\.010565\)", "+42.011"),
        (r"^\(\-17\.027\)", "-17.027"),
        (r"^\(\-17\.026549\)", "-17.027"),
    ]
    for pattern, replacement in mass_mappings:
        peptide = re.sub(pattern, replacement, peptide)
    return peptide


def normalize_ptm_format(peptide: str, model: str | None = None) -> str:
    """Normalize model-specific PTM strings into a shared mass-offset form."""
    if not peptide:
        return ""

    if model == "instanovo":
        peptide = convert_mass_offset_brackets_to_standard(peptide)
        peptide = convert_unimod_to_mass_format(peptide)
    elif model == "casanovo":
        peptide = convert_mass_offset_brackets_to_standard(peptide)
        peptide = convert_casanovo_to_mass_format(peptide)
    elif model == "primenovo" or model is None:
        peptide = convert_mass_offset_brackets_to_standard(peptide)
    elif model == "ultraprot":
        peptide = convert_ultraprot_to_mass_format(peptide)

    return peptide.replace("I", "L")


def remove_c_modifications(peptide: str) -> str:
    if not peptide:
        return peptide

    peptide = re.sub(r"C\+57\.021", "C", peptide)
    peptide = re.sub(r"C\(carbamidomethyl\)", "C", peptide)
    peptide = re.sub(r"C\[UNIMOD:4\]", "C", peptide)
    return peptide


def parse_peptide_with_ptm(peptide: str) -> List[Dict[str, object]]:
    """Parse a peptide into residue/PTM records."""
    if not peptide:
        return []

    residues: List[Dict[str, object]] = []
    peptide = re.sub(r"C\+57\.021", "C(carbamidomethyl)", peptide)
    peptide = re.sub(r"C\+57\.02146", "C(carbamidomethyl)", peptide)
    peptide = re.sub(r"Q\+0\.984", "Q(deamide)", peptide)
    peptide = re.sub(r"Q\+0\.98401", "Q(deamide)", peptide)
    peptide = re.sub(r"N\+0\.984", "N(deamide)", peptide)
    peptide = re.sub(r"N\+0\.98401", "N(deamide)", peptide)
    peptide = re.sub(r"M\+15\.995", "M(ox)", peptide)
    peptide = re.sub(r"M\+15\.99491", "M(ox)", peptide)
    peptide = re.sub(r"S\+79\.966", "S(phos)", peptide)
    peptide = re.sub(r"T\+79\.966", "T(phos)", peptide)
    peptide = re.sub(r"Y\+79\.966", "Y(phos)", peptide)

    prefix_mass_match = re.match(r"^([+-]\d+\.\d+)", peptide)
    if prefix_mass_match:
        mass_str = prefix_mass_match.group(1)
        if mass_str in {"+42.011", "+42.010565"}:
            residues.append({"aa": "*", "ptm": "acetyl", "has_ptm": True})
        elif mass_str in {"-17.027", "-17.026549"}:
            residues.append({"aa": "*", "ptm": "-nh3", "has_ptm": True})
        else:
            residues.append({"aa": "*", "ptm": mass_str, "has_ptm": True})
        peptide = peptide[prefix_mass_match.end() :]

    prefix_match = re.match(r"^\((acetyl|carbamyl|-nh3|carbamyl\)\(-nh3)\)", peptide)
    if prefix_match:
        residues.append({"aa": "*", "ptm": prefix_match.group(1), "has_ptm": True})
        remaining = peptide[prefix_match.end() :]
    else:
        remaining = peptide

    for match in re.finditer(r"([A-Z])(?:\(([^)]+)\))?", remaining):
        aa = match.group(1)
        ptm = match.group(2)
        residues.append({"aa": aa, "ptm": ptm.strip() if ptm else None, "has_ptm": bool(ptm)})

    return residues


def calculate_residue_mass(residue: Dict[str, object]) -> float:
    base_mass = AMINO_ACID_MASSES.get(str(residue["aa"]), 0.0)
    if residue.get("has_ptm") and residue.get("ptm"):
        base_mass += PTM_MASSES.get(str(residue["ptm"]), 0.0)
    return base_mass


def calculate_sequence_masses(residues: List[Dict[str, object]]) -> Tuple[List[float], float]:
    prefix_masses = [0.0]
    total_mass = 0.0
    for residue in residues:
        total_mass += calculate_residue_mass(residue)
        prefix_masses.append(total_mass)
    return prefix_masses, total_mass


def truncate_prediction(
    pred_residues: List[Dict[str, object]],
    pred_prefix_masses: List[float],
    true_total_mass: float,
) -> Tuple[List[Dict[str, object]], List[float]]:
    """Keep compatibility with older rerank logic that trims overlong candidates."""
    pred_total = pred_prefix_masses[-1] if pred_prefix_masses else 0.0
    if pred_total <= true_total_mass + 2:
        return pred_residues, pred_prefix_masses

    truncate_idx = bisect.bisect_right(pred_prefix_masses, true_total_mass) - 1
    truncate_idx = max(truncate_idx, 0)
    return pred_residues[:truncate_idx], pred_prefix_masses[: truncate_idx + 1]
