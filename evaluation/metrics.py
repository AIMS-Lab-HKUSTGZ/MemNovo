"""
Evaluation Metrics

Functions for computing de novo peptide sequencing metrics.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


def normalize_sequence(sequence: str) -> str:
    """
    Normalize peptide sequence for comparison.

    - Convert to uppercase
    - Replace I with L (isobaric equivalence)
    - Remove modification markers and non-alphabetic characters

    Args:
        sequence: Raw peptide sequence

    Returns:
        Normalized sequence
    """
    if not sequence:
        return ''

    # Uppercase
    seq = sequence.upper()

    # I/L equivalence
    seq = seq.replace('I', 'L')

    # Keep only letters
    seq = ''.join(c for c in seq if c.isalpha())

    return seq


def compute_aa_match(predicted: str, target: str) -> int:
    """
    Compute amino acid level matches using frequency-based counting.

    This method counts the minimum frequency of each amino acid
    in both sequences, similar to the Novor algorithm.

    Args:
        predicted: Predicted sequence (normalized)
        target: Target sequence (normalized)

    Returns:
        Number of matched amino acids
    """
    if not predicted or not target:
        return 0

    n_match = 0
    all_aas = set(predicted + target)

    for aa in all_aas:
        n_match += min(predicted.count(aa), target.count(aa))

    return n_match


def aa_precision(predicted: str, target: str, normalize: bool = True) -> float:
    """
    Compute amino acid precision.

    Precision = matched_aa / predicted_aa

    Args:
        predicted: Predicted sequence
        target: Target sequence
        normalize: Whether to normalize sequences

    Returns:
        Precision score (0-1)
    """
    if normalize:
        predicted = normalize_sequence(predicted)
        target = normalize_sequence(target)

    if not predicted:
        return 0.0

    n_match = compute_aa_match(predicted, target)
    return n_match / len(predicted)


def aa_recall(predicted: str, target: str, normalize: bool = True) -> float:
    """
    Compute amino acid recall.

    Recall = matched_aa / target_aa

    Args:
        predicted: Predicted sequence
        target: Target sequence
        normalize: Whether to normalize sequences

    Returns:
        Recall score (0-1)
    """
    if normalize:
        predicted = normalize_sequence(predicted)
        target = normalize_sequence(target)

    if not target:
        return 0.0

    n_match = compute_aa_match(predicted, target)
    return n_match / len(target)


def peptide_precision(predicted: str, target: str, normalize: bool = True) -> float:
    """
    Compute peptide-level precision (exact match).

    Args:
        predicted: Predicted sequence
        target: Target sequence
        normalize: Whether to normalize sequences

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if normalize:
        predicted = normalize_sequence(predicted)
        target = normalize_sequence(target)

    return 1.0 if predicted == target else 0.0


def peptide_recall(predicted: str, target: str, normalize: bool = True) -> float:
    """
    Compute peptide-level recall (same as precision for single sample).

    Args:
        predicted: Predicted sequence
        target: Target sequence
        normalize: Whether to normalize sequences

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return peptide_precision(predicted, target, normalize)


def compute_all_metrics(
    predicted: str,
    target: str,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single prediction.

    Args:
        predicted: Predicted sequence
        target: Target sequence
        normalize: Whether to normalize sequences

    Returns:
        Dictionary with all metrics
    """
    if normalize:
        predicted = normalize_sequence(predicted)
        target = normalize_sequence(target)

    n_match = compute_aa_match(predicted, target)
    n_pred = len(predicted)
    n_target = len(target)

    return {
        'aa_precision': n_match / n_pred if n_pred > 0 else 0.0,
        'aa_recall': n_match / n_target if n_target > 0 else 0.0,
        'peptide_match': 1.0 if predicted == target else 0.0,
        'n_match_aa': n_match,
        'n_pred_aa': n_pred,
        'n_target_aa': n_target,
    }


def aggregate_metrics(
    predictions: List[str],
    targets: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Aggregate metrics over multiple samples.

    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
        normalize: Whether to normalize sequences

    Returns:
        Aggregated metrics dictionary
    """
    n_samples = len(predictions)
    n_target_aa = 0
    n_pred_aa = 0
    n_match_aa = 0
    n_match_pep = 0

    for pred, target in zip(predictions, targets):
        if normalize:
            pred = normalize_sequence(pred)
            target = normalize_sequence(target)

        n_target_aa += len(target)
        n_pred_aa += len(pred)
        n_match_aa += compute_aa_match(pred, target)

        if pred == target:
            n_match_pep += 1

    return {
        'aa_precision': n_match_aa / n_pred_aa if n_pred_aa > 0 else 0.0,
        'aa_recall': n_match_aa / n_target_aa if n_target_aa > 0 else 0.0,
        'pep_precision': n_match_pep / n_samples if n_samples > 0 else 0.0,
        'pep_recall': n_match_pep / n_samples if n_samples > 0 else 0.0,
        'n_samples': n_samples,
        'n_match_pep': n_match_pep,
        'n_match_aa': n_match_aa,
        'n_pred_aa': n_pred_aa,
        'n_target_aa': n_target_aa,
    }
