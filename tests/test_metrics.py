"""
Tests for evaluation metrics module.
"""

import pytest
from evaluation.metrics import (
    normalize_sequence,
    compute_aa_match,
    aa_precision,
    aa_recall,
    peptide_precision,
    compute_all_metrics,
    aggregate_metrics,
)


class TestNormalizeSequence:
    """Tests for sequence normalization."""

    def test_uppercase(self):
        """Test conversion to uppercase."""
        assert normalize_sequence('peptide') == 'PEPTLDE'

    def test_il_equivalence(self):
        """Test I/L equivalence."""
        assert normalize_sequence('PEPTIDE') == 'PEPTLDE'
        assert normalize_sequence('LEUCINE') == 'LEUCLNE'

    def test_remove_modifications(self):
        """Test removal of non-alphabetic characters."""
        assert normalize_sequence('PEP[+15.99]TIDE') == 'PEPTLDE'
        assert normalize_sequence('M*PEPTIDE') == 'MPEPTLDE'

    def test_empty_string(self):
        """Test empty input."""
        assert normalize_sequence('') == ''


class TestAAMatch:
    """Tests for amino acid matching."""

    def test_exact_match(self):
        """Test exact sequence match."""
        assert compute_aa_match('PEPTIDE', 'PEPTIDE') == 7

    def test_partial_match(self):
        """Test partial match."""
        assert compute_aa_match('PEPTIDE', 'PEPTXXX') == 4

    def test_no_match(self):
        """Test no common amino acids."""
        assert compute_aa_match('AAA', 'BBB') == 0

    def test_frequency_based(self):
        """Test frequency-based matching."""
        # AAAB and AABB: A matches twice, B matches once = 3
        assert compute_aa_match('AAAB', 'AABB') == 3


class TestAAPrecision:
    """Tests for AA precision calculation."""

    def test_perfect_precision(self):
        """Test 100% precision."""
        assert aa_precision('PEPTIDE', 'PEPTIDE') == 1.0

    def test_partial_precision(self):
        """Test partial precision."""
        # PEPT matches 4 out of 4 predicted
        result = aa_precision('PEPT', 'PEPTIDE')
        assert result == 1.0

    def test_empty_prediction(self):
        """Test empty prediction."""
        assert aa_precision('', 'PEPTIDE') == 0.0


class TestAARecall:
    """Tests for AA recall calculation."""

    def test_perfect_recall(self):
        """Test 100% recall."""
        assert aa_recall('PEPTIDE', 'PEPTIDE') == 1.0

    def test_partial_recall(self):
        """Test partial recall."""
        # PEPT matches 4 out of 7 target
        result = aa_recall('PEPT', 'PEPTIDE')
        assert result == pytest.approx(4/7, rel=0.01)

    def test_empty_target(self):
        """Test empty target."""
        assert aa_recall('PEPTIDE', '') == 0.0


class TestPeptidePrecision:
    """Tests for peptide-level precision."""

    def test_exact_match(self):
        """Test exact match."""
        assert peptide_precision('PEPTIDE', 'PEPTIDE') == 1.0

    def test_mismatch(self):
        """Test mismatch."""
        assert peptide_precision('PEPTIDE', 'PEPTIDX') == 0.0

    def test_il_equivalence(self):
        """Test I/L equivalence in matching."""
        assert peptide_precision('PEPTIDE', 'PEPTLDE') == 1.0


class TestAggregateMetrics:
    """Tests for metric aggregation."""

    def test_perfect_predictions(self):
        """Test all perfect predictions."""
        predictions = ['PEPTIDE', 'SEQUENCE', 'PROTEIN']
        targets = ['PEPTIDE', 'SEQUENCE', 'PROTEIN']

        metrics = aggregate_metrics(predictions, targets)

        assert metrics['aa_precision'] == 1.0
        assert metrics['aa_recall'] == 1.0
        assert metrics['pep_precision'] == 1.0

    def test_mixed_predictions(self):
        """Test mix of correct and incorrect."""
        predictions = ['PEPTIDE', 'WRONGXX']
        targets = ['PEPTIDE', 'CORRECT']

        metrics = aggregate_metrics(predictions, targets)

        assert metrics['n_samples'] == 2
        assert metrics['n_match_pep'] == 1
        assert 0 < metrics['aa_precision'] < 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
