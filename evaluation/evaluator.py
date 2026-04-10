"""
Paper-oriented evaluation utilities.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from memnovo.backends import ensure_external_imports

from .metrics import compute_aa_match, normalize_sequence

logger = logging.getLogger(__name__)


def _strip_common_ptm_markup(sequence: str) -> str:
    """Normalize common peptide string markup to a plain AA sequence."""
    if not sequence:
        return ""

    seq = sequence.upper()
    seq = re.sub(r"<[^>]*>", "", seq)
    seq = re.sub(r"\[[^\]]*\]", "", seq)
    seq = re.sub(r"\([^)]+\)", "", seq)
    seq = re.sub(r"[+-]\d+(?:\.\d+)?", "", seq)
    return seq.replace("-", "")


def _sanitize_instanovo_sequence_for_metrics(sequence: str) -> str:
    """Map archive/public PTM spellings onto tokens accepted by InstaNovo Metrics.

    The upstream Metrics splitter ignores valid N-term tokens at sequence start, but
    some archived predictions contain malformed residue-level forms such as
    ``D[UNIMOD:5]``. Those should be treated as a plain residue for evaluation rather
    than crashing residue mass lookup.
    """
    if not sequence:
        return ""

    seq = sequence
    seq = seq.replace("M+15.995", "M[UNIMOD:35]")
    seq = seq.replace("M+15.99491", "M[UNIMOD:35]")
    seq = re.sub(r"^\+42\.011", "", seq)
    seq = re.sub(r"^\+42\.010565", "", seq)
    seq = re.sub(r"^\+43\.006", "", seq)
    seq = re.sub(r"^\+43\.005814", "", seq)
    seq = re.sub(r"^-17\.027", "", seq)
    seq = re.sub(r"^-17\.026549", "", seq)
    seq = re.sub(r"\[UNIMOD:(?:1|5|385)\]", "", seq)
    return seq


class Evaluator:
    """Unified evaluator with official Casanovo and InstaNovo backends."""

    def __init__(self, mass_tolerance: float = 50.0, normalize_il: bool = True):
        self.mass_tolerance = mass_tolerance
        self.normalize_il = normalize_il

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        model_name: Optional[str] = None,
    ) -> Dict[str, float]:
        model_name = (model_name or "").lower()
        if model_name.startswith("casanovo"):
            return self.evaluate_casanovo(predictions, ground_truth)
        if model_name.startswith("instanovo"):
            return self.evaluate_instanovo(predictions, ground_truth)
        return self.evaluate_generic(predictions, ground_truth)

    def evaluate_generic(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        n_samples = min(len(predictions), len(ground_truth))
        n_target_aa = 0
        n_pred_aa = 0
        n_match_aa = 0
        n_pred_pep = 0
        n_match_pep = 0

        for pred, truth in zip(predictions[:n_samples], ground_truth[:n_samples]):
            pred_seq = pred.get("sequence", "")
            truth_seq = truth.get("sequence", "")
            if self.normalize_il:
                pred_seq = normalize_sequence(pred_seq)
                truth_seq = normalize_sequence(truth_seq)

            n_target_aa += len(truth_seq)
            if pred_seq:
                n_pred_pep += 1
                n_pred_aa += len(pred_seq)
                n_match_aa += compute_aa_match(pred_seq, truth_seq)
                if pred_seq == truth_seq:
                    n_match_pep += 1

        return self._finalize_metrics(
            aa_precision=n_match_aa / n_pred_aa if n_pred_aa else 0.0,
            aa_recall=n_match_aa / n_target_aa if n_target_aa else 0.0,
            pep_precision=n_match_pep / n_pred_pep if n_pred_pep else 0.0,
            pep_recall=n_match_pep / n_samples if n_samples else 0.0,
            total_samples=n_samples,
            valid_predictions=n_pred_pep,
            matched_peptides=n_match_pep,
            method="generic",
            n_aa_true=n_target_aa,
            n_aa_pred=n_pred_aa,
            n_aa_correct=n_match_aa,
            n_pred_pep=n_pred_pep,
            n_match_pep=n_match_pep,
        )

    def evaluate_casanovo(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        ensure_external_imports()
        from casanovo.denovo.evaluate import aa_match_batch, aa_match_metrics

        aa_masses = {
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "C": 160.030649,
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

        pred_list = []
        truth_list = []
        for pred, truth in zip(predictions, ground_truth):
            pred_seq = _strip_common_ptm_markup(pred.get("sequence", ""))
            truth_seq = _strip_common_ptm_markup(truth.get("sequence", ""))
            if pred_seq and truth_seq:
                if self.normalize_il:
                    pred_seq = normalize_sequence(pred_seq)
                    truth_seq = normalize_sequence(truth_seq)
                pred_list.append(pred_seq)
                truth_list.append(truth_seq)

        if not pred_list:
            raise ValueError("No valid Casanovo predictions available for evaluation")

        aa_matches_batch, n_aa_true, n_aa_pred = aa_match_batch(truth_list, pred_list, aa_masses)
        aa_precision, aa_recall, pep_precision = aa_match_metrics(
            aa_matches_batch,
            n_aa_true,
            n_aa_pred,
        )
        n_aa_correct = int(sum(aa_matches[0].sum() for aa_matches in aa_matches_batch))
        n_match_pep = int(sum(int(bool(aa_matches[1])) for aa_matches in aa_matches_batch))

        return self._finalize_metrics(
            aa_precision=float(aa_precision),
            aa_recall=float(aa_recall),
            pep_precision=float(pep_precision),
            pep_recall=float(n_match_pep / len(predictions) if predictions else 0.0),
            total_samples=len(predictions),
            valid_predictions=len(pred_list),
            matched_peptides=n_match_pep,
            method="casanovo_official",
            n_aa_true=int(n_aa_true),
            n_aa_pred=int(n_aa_pred),
            n_aa_correct=n_aa_correct,
            n_pred_pep=len(pred_list),
            n_match_pep=n_match_pep,
        )

    def evaluate_instanovo(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        ensure_external_imports()
        from instanovo.utils.metrics import Metrics
        from instanovo.utils.residues import ResidueSet

        residue_masses = {
            "A": 71.037114,
            "R": 156.101111,
            "N": 114.042927,
            "D": 115.026943,
            "C": 103.009185,
            "E": 129.042593,
            "Q": 128.058578,
            "G": 57.021464,
            "H": 137.058912,
            "I": 113.084064,
            "L": 113.084064,
            "K": 128.094963,
            "M": 131.040485,
            "F": 147.068414,
            "P": 97.052764,
            "S": 87.032028,
            "T": 101.047670,
            "W": 186.079313,
            "Y": 163.063329,
            "V": 99.068414,
            "M+15.995": 147.035400,
            "C+57.021": 160.030649,
            "N+0.984": 115.026943,
            "Q+0.984": 129.042594,
            "M[UNIMOD:35]": 147.035400,
            "C[UNIMOD:4]": 160.030649,
            "N[UNIMOD:7]": 115.026943,
            "Q[UNIMOD:7]": 129.042594,
            "S[UNIMOD:21]": 166.998028,
            "T[UNIMOD:21]": 181.013670,
            "Y[UNIMOD:21]": 243.029329,
            "[UNIMOD:1]": 42.010565,
            "[UNIMOD:5]": 43.005814,
            "[UNIMOD:385]": -17.026549,
        }

        metrics_calc = Metrics(residue_set=ResidueSet(residue_masses), isotope_error_range=[0, 1])
        targets = [_sanitize_instanovo_sequence_for_metrics(truth.get("sequence", "")) for truth in ground_truth]
        preds = [_sanitize_instanovo_sequence_for_metrics(pred.get("sequence", "")) for pred in predictions]

        if len(targets) != len(preds):
            raise ValueError(
                f"Prediction/target length mismatch for InstaNovo evaluation: {len(preds)} vs {len(targets)}"
            )

        n_targ_aa, n_pred_aa, n_match_aa = 0, 0, 0
        n_pred_pep, n_match_pep = 0, 0
        for target, pred in zip(targets, preds):
            targ_split = metrics_calc._split_peptide(target)
            pred_split = metrics_calc._split_peptide(pred)
            n_targ_aa += len(targ_split)
            if len(pred_split) > 0:
                n_pred_aa += len(pred_split)
                n_pred_pep += 1
                n_match = metrics_calc._novor_match(targ_split, pred_split)
                n_match_aa += n_match
                if len(pred_split) == len(targ_split) and len(targ_split) == n_match:
                    n_match_pep += 1

        aa_precision, aa_recall, pep_recall, pep_precision = metrics_calc.compute_precision_recall(
            targets=targets,
            predictions=preds,
        )

        return self._finalize_metrics(
            aa_precision=float(aa_precision),
            aa_recall=float(aa_recall),
            pep_precision=float(pep_precision),
            pep_recall=float(pep_recall),
            total_samples=len(targets),
            valid_predictions=len(targets),
            method="instanovo_official",
            n_aa_true=n_targ_aa,
            n_aa_pred=n_pred_aa,
            n_aa_correct=n_match_aa,
            n_pred_pep=n_pred_pep,
            n_match_pep=n_match_pep,
        )

    def evaluate_by_length(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        length_bins: Optional[List[tuple[int, int]]] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        if length_bins is None:
            length_bins = [(7, 10), (11, 15), (16, 20), (21, 10**9)]

        groups = defaultdict(lambda: {"pred": [], "truth": []})
        for pred, truth in zip(predictions, ground_truth):
            length = len(normalize_sequence(truth.get("sequence", "")))
            for lower, upper in length_bins:
                if lower <= length <= upper:
                    key = f"{lower}-{upper if upper < 10**9 else 'plus'}"
                    groups[key]["pred"].append(pred)
                    groups[key]["truth"].append(truth)
                    break

        return {
            key: self.evaluate(value["pred"], value["truth"], model_name=model_name)
            for key, value in groups.items()
            if value["pred"]
        }

    @staticmethod
    def _finalize_metrics(**metrics: float) -> Dict[str, float]:
        finalized: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, str):
                finalized[key] = value
                continue
            if isinstance(value, bool):
                finalized[key] = float(value)
            elif isinstance(value, int):
                finalized[key] = int(value)
            else:
                finalized[key] = float(value)
        return finalized


def aggregate_metrics(metric_dicts: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate shard-level metrics exactly from additive raw counts."""
    if not metric_dicts:
        raise ValueError("No metric dictionaries provided for aggregation")

    method = str(metric_dicts[0].get("method", "generic"))
    total_samples = int(sum(int(item.get("total_samples", 0)) for item in metric_dicts))
    n_aa_true = int(sum(int(item.get("n_aa_true", 0)) for item in metric_dicts))
    n_aa_pred = int(sum(int(item.get("n_aa_pred", 0)) for item in metric_dicts))
    n_aa_correct = int(sum(int(item.get("n_aa_correct", 0)) for item in metric_dicts))
    n_pred_pep = int(sum(int(item.get("n_pred_pep", item.get("valid_predictions", 0))) for item in metric_dicts))
    n_match_pep = int(sum(int(item.get("n_match_pep", item.get("matched_peptides", 0))) for item in metric_dicts))

    aa_precision = n_aa_correct / n_aa_pred if n_aa_pred else 0.0
    aa_recall = n_aa_correct / n_aa_true if n_aa_true else 0.0
    pep_precision = n_match_pep / n_pred_pep if n_pred_pep else 0.0
    pep_recall = n_match_pep / total_samples if total_samples else 0.0

    return {
        "aa_precision": float(aa_precision),
        "aa_recall": float(aa_recall),
        "pep_precision": float(pep_precision),
        "pep_recall": float(pep_recall),
        "total_samples": total_samples,
        "valid_predictions": n_pred_pep,
        "matched_peptides": n_match_pep,
        "method": method,
        "n_aa_true": n_aa_true,
        "n_aa_pred": n_aa_pred,
        "n_aa_correct": n_aa_correct,
        "n_pred_pep": n_pred_pep,
        "n_match_pep": n_match_pep,
    }


def compute_metrics(
    predictions: List[str],
    targets: List[str],
    normalize_il: bool = True,
) -> Dict[str, float]:
    pred_dicts = [{"sequence": seq} for seq in predictions]
    target_dicts = [{"sequence": seq} for seq in targets]
    evaluator = Evaluator(normalize_il=normalize_il)
    return evaluator.evaluate_generic(pred_dicts, target_dicts)
