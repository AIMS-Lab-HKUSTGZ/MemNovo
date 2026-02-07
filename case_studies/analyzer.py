"""
Case Study Analyzer

Tools for investigating individual predictions where MemNovo
corrects baseline model errors.
"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class CaseStudy:
    """Individual case study result."""

    spectrum_id: str
    target: str
    baseline_pred: str
    memnovo_pred: str
    baseline_correct: bool
    memnovo_correct: bool
    improvement_type: str  # 'correction', 'degradation', 'unchanged'

    # Optional metadata
    precursor_mz: Optional[float] = None
    charge: Optional[int] = None
    species: Optional[str] = None


def identify_corrections(
    baseline_predictions: list[str],
    memnovo_predictions: list[str],
    targets: list[str],
    spectrum_ids: list[str],
    metadata: Optional[list[dict]] = None
) -> list[CaseStudy]:
    """
    Identify cases where MemNovo corrects baseline errors.

    Parameters
    ----------
    baseline_predictions : list[str]
        Baseline model predictions.
    memnovo_predictions : list[str]
        MemNovo-enhanced predictions.
    targets : list[str]
        Ground truth sequences.
    spectrum_ids : list[str]
        Spectrum identifiers.
    metadata : list[dict], optional
        Additional metadata per spectrum.

    Returns
    -------
    list[CaseStudy]
        List of case studies where MemNovo made a correction.
    """
    from evaluation.metrics import normalize_sequence

    corrections = []
    metadata = metadata or [{}] * len(targets)

    for i, (baseline, memnovo, target, sid, meta) in enumerate(
        zip(baseline_predictions, memnovo_predictions, targets, spectrum_ids, metadata)
    ):
        baseline_norm = normalize_sequence(baseline)
        memnovo_norm = normalize_sequence(memnovo)
        target_norm = normalize_sequence(target)

        baseline_correct = baseline_norm == target_norm
        memnovo_correct = memnovo_norm == target_norm

        if not baseline_correct and memnovo_correct:
            improvement_type = 'correction'
        elif baseline_correct and not memnovo_correct:
            improvement_type = 'degradation'
        else:
            improvement_type = 'unchanged'

        if improvement_type == 'correction':
            corrections.append(CaseStudy(
                spectrum_id=sid,
                target=target,
                baseline_pred=baseline,
                memnovo_pred=memnovo,
                baseline_correct=baseline_correct,
                memnovo_correct=memnovo_correct,
                improvement_type=improvement_type,
                precursor_mz=meta.get('precursor_mz'),
                charge=meta.get('charge'),
                species=meta.get('species')
            ))

    return corrections


def analyze_error_patterns(
    corrections: list[CaseStudy]
) -> dict:
    """
    Analyze patterns in corrected errors.

    Returns
    -------
    dict
        Analysis results including:
        - Common substitution patterns
        - Length distributions
        - Position-based error rates
    """
    from collections import Counter

    substitutions = Counter()
    length_dist = Counter()
    error_positions = Counter()

    for case in corrections:
        baseline = case.baseline_pred
        target = case.target

        # Track length
        length_dist[len(target)] += 1

        # Find substitution positions
        min_len = min(len(baseline), len(target))
        for i in range(min_len):
            if baseline[i] != target[i]:
                substitutions[f"{baseline[i]}->{target[i]}"] += 1
                # Normalize position to 0-1 range
                norm_pos = i / len(target)
                pos_bin = int(norm_pos * 10) / 10
                error_positions[pos_bin] += 1

    return {
        'n_corrections': len(corrections),
        'top_substitutions': substitutions.most_common(10),
        'length_distribution': dict(length_dist),
        'error_position_distribution': dict(error_positions)
    }


def export_case_studies(
    corrections: list[CaseStudy],
    output_path: str,
    format: str = 'json'
) -> None:
    """
    Export case studies to file.

    Parameters
    ----------
    corrections : list[CaseStudy]
        Case studies to export.
    output_path : str
        Output file path.
    format : str
        Export format ('json' or 'csv').
    """
    if format == 'json':
        data = [
            {
                'spectrum_id': c.spectrum_id,
                'target': c.target,
                'baseline_pred': c.baseline_pred,
                'memnovo_pred': c.memnovo_pred,
                'improvement_type': c.improvement_type,
                'precursor_mz': c.precursor_mz,
                'charge': c.charge,
                'species': c.species
            }
            for c in corrections
        ]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    elif format == 'csv':
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'spectrum_id', 'target', 'baseline_pred',
                'memnovo_pred', 'improvement_type'
            ])
            for c in corrections:
                writer.writerow([
                    c.spectrum_id, c.target, c.baseline_pred,
                    c.memnovo_pred, c.improvement_type
                ])


if __name__ == '__main__':
    # Example usage
    baseline = ['PEPTIDE', 'WRONGXX', 'CORRECT']
    memnovo = ['PEPTIDE', 'CORRECT', 'CORRECT']
    targets = ['PEPTIDE', 'CORRECT', 'CORRECT']
    ids = ['spec_1', 'spec_2', 'spec_3']

    corrections = identify_corrections(baseline, memnovo, targets, ids)
    print(f"Found {len(corrections)} corrections")

    analysis = analyze_error_patterns(corrections)
    print(f"Analysis: {analysis}")
