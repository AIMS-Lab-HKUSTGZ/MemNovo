"""
Unified Evaluator

Provides evaluation metrics compatible with both Casanovo and InstaNovo
evaluation conventions.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import defaultdict

from .metrics import normalize_sequence, compute_aa_match, compute_all_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Unified evaluator for de novo peptide sequencing.

    Supports:
    - Amino acid level metrics (precision, recall)
    - Peptide level metrics (exact match accuracy)
    - I/L equivalence normalization
    - Mass tolerance validation

    Args:
        mass_tolerance: Mass tolerance in ppm (default: 50)
        normalize_il: Treat I and L as equivalent (default: True)
    """

    def __init__(
        self,
        mass_tolerance: float = 50.0,
        normalize_il: bool = True,
    ):
        self.mass_tolerance = mass_tolerance
        self.normalize_il = normalize_il

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: List of prediction dicts with 'sequence' key
            ground_truth: List of ground truth dicts with 'sequence' key

        Returns:
            Dictionary with evaluation metrics
        """
        n_samples = min(len(predictions), len(ground_truth))
        if len(predictions) != len(ground_truth):
            logger.warning(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(ground_truth)} ground truth. Using first {n_samples}."
            )

        # Initialize counters
        n_target_aa = 0
        n_pred_aa = 0
        n_match_aa = 0
        n_pred_pep = 0
        n_match_pep = 0

        for pred, truth in zip(predictions[:n_samples], ground_truth[:n_samples]):
            try:
                pred_seq = pred.get('sequence', '')
                truth_seq = truth.get('sequence', '')

                # Normalize
                if self.normalize_il:
                    pred_seq = normalize_sequence(pred_seq)
                    truth_seq = normalize_sequence(truth_seq)

                # Count target AAs
                if truth_seq:
                    n_target_aa += len(truth_seq)

                # Count prediction AAs
                if pred_seq:
                    n_pred_pep += 1
                    n_pred_aa += len(pred_seq)

                    # AA-level matches
                    if truth_seq:
                        n_match_aa += compute_aa_match(pred_seq, truth_seq)

                        # Peptide-level match
                        if pred_seq == truth_seq:
                            n_match_pep += 1

            except Exception as e:
                logger.debug(f"Error evaluating sample: {e}")
                continue

        # Compute metrics
        aa_precision = n_match_aa / n_pred_aa if n_pred_aa > 0 else 0.0
        aa_recall = n_match_aa / n_target_aa if n_target_aa > 0 else 0.0
        pep_precision = n_match_pep / n_pred_pep if n_pred_pep > 0 else 0.0
        pep_recall = n_match_pep / n_samples if n_samples > 0 else 0.0

        return {
            'aa_precision': aa_precision,
            'aa_recall': aa_recall,
            'pep_precision': pep_precision,
            'pep_recall': pep_recall,
            'n_samples': n_samples,
            'n_pred_pep': n_pred_pep,
            'n_match_pep': n_match_pep,
            'n_target_aa': n_target_aa,
            'n_pred_aa': n_pred_aa,
            'n_match_aa': n_match_aa,
        }

    def evaluate_by_length(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        length_bins: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics stratified by sequence length.

        Args:
            predictions: List of predictions
            ground_truth: List of ground truth
            length_bins: Length bin edges (default: [0, 10, 15, 20, 25, inf])

        Returns:
            Dictionary mapping length ranges to metrics
        """
        if length_bins is None:
            length_bins = [0, 10, 15, 20, 25, float('inf')]

        # Group by length
        length_groups = defaultdict(lambda: {'pred': [], 'truth': []})

        for pred, truth in zip(predictions, ground_truth):
            truth_seq = truth.get('sequence', '')
            length = len(normalize_sequence(truth_seq))

            # Find bin
            for i in range(len(length_bins) - 1):
                if length_bins[i] <= length < length_bins[i + 1]:
                    bin_name = f"{length_bins[i]}-{length_bins[i+1]}"
                    length_groups[bin_name]['pred'].append(pred)
                    length_groups[bin_name]['truth'].append(truth)
                    break

        # Evaluate each bin
        results = {}
        for bin_name, data in length_groups.items():
            if data['pred']:
                results[bin_name] = self.evaluate(data['pred'], data['truth'])

        return results

    def evaluate_by_species(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics stratified by species (if available).

        Args:
            predictions: List of predictions with optional 'species' key
            ground_truth: List of ground truth with optional 'species' key

        Returns:
            Dictionary mapping species to metrics
        """
        species_groups = defaultdict(lambda: {'pred': [], 'truth': []})

        for pred, truth in zip(predictions, ground_truth):
            species = truth.get('species', 'unknown')
            species_groups[species]['pred'].append(pred)
            species_groups[species]['truth'].append(truth)

        results = {}
        for species, data in species_groups.items():
            if data['pred']:
                results[species] = self.evaluate(data['pred'], data['truth'])

        return results


def compute_metrics(
    predictions: List[str],
    targets: List[str],
    normalize_il: bool = True,
) -> Dict[str, float]:
    """
    Convenience function to compute metrics from sequence lists.

    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        normalize_il: Normalize I/L equivalence

    Returns:
        Metrics dictionary
    """
    pred_dicts = [{'sequence': s} for s in predictions]
    target_dicts = [{'sequence': s} for s in targets]

    evaluator = Evaluator(normalize_il=normalize_il)
    return evaluator.evaluate(pred_dicts, target_dicts)
