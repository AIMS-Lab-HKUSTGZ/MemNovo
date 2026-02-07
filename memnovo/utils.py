"""
MemNovo Utility Functions

Common utilities for data handling, logging, and configuration.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Output path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {config_path}")


def load_spectra(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load spectra from MGF file.

    Args:
        path: Path to MGF file

    Returns:
        List of spectrum dictionaries
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {path}")

    spectra = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        spectrum = {}
        mz_list = []
        intensity_list = []

        for line in f:
            line = line.strip()

            if line.startswith("BEGIN IONS"):
                spectrum = {}
                mz_list = []
                intensity_list = []

            elif line.startswith("PEPMASS="):
                parts = line.replace("PEPMASS=", "").split()
                spectrum["precursor_mz"] = float(parts[0])

            elif line.startswith("CHARGE="):
                charge_str = line.replace("CHARGE=", "").strip()
                spectrum["precursor_charge"] = int(charge_str.rstrip("+-"))

            elif line.startswith("TITLE="):
                spectrum["spectrum_id"] = line.replace("TITLE=", "").strip()

            elif line.startswith("SEQ="):
                spectrum["sequence"] = line.replace("SEQ=", "").strip()

            elif line.startswith("END IONS"):
                if mz_list:
                    spectrum["mz_array"] = np.array(mz_list, dtype=np.float32)
                    spectrum["intensity_array"] = np.array(intensity_list, dtype=np.float32)
                    spectra.append(spectrum)

            elif line and not line.startswith(('BEGIN', 'END', 'TITLE', 'PEPMASS', 'CHARGE', 'SEQ', '=')):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz_list.append(float(parts[0]))
                        intensity_list.append(float(parts[1]))
                except (ValueError, IndexError):
                    pass

    logger.info(f"Loaded {len(spectra)} spectra from {path}")
    return spectra


def normalize_sequence(sequence: str) -> str:
    """
    Normalize peptide sequence for comparison.

    - Converts to uppercase
    - Replaces I with L (isobaric)
    - Removes modification markers

    Args:
        sequence: Raw peptide sequence

    Returns:
        Normalized sequence
    """
    normalized = sequence.upper()
    normalized = normalized.replace('I', 'L')
    normalized = ''.join(c for c in normalized if c.isalpha())
    return normalized


def compute_aa_accuracy(
    predicted: str,
    target: str,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute amino acid level accuracy metrics.

    Args:
        predicted: Predicted peptide sequence
        target: Ground truth sequence
        normalize: Whether to normalize sequences

    Returns:
        Dictionary with precision, recall, and match count
    """
    if normalize:
        predicted = normalize_sequence(predicted)
        target = normalize_sequence(target)

    # Count matches based on character frequency
    n_match = 0
    for aa in set(predicted + target):
        n_match += min(predicted.count(aa), target.count(aa))

    n_pred = len(predicted)
    n_target = len(target)

    precision = n_match / n_pred if n_pred > 0 else 0.0
    recall = n_match / n_target if n_target > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'n_match': n_match,
        'n_pred': n_pred,
        'n_target': n_target,
    }


def compute_peptide_accuracy(
    predicted: str,
    target: str,
    normalize: bool = True,
) -> bool:
    """
    Check if predicted peptide exactly matches target.

    Args:
        predicted: Predicted peptide sequence
        target: Ground truth sequence
        normalize: Whether to normalize sequences

    Returns:
        True if sequences match exactly
    """
    if normalize:
        predicted = normalize_sequence(predicted)
        target = normalize_sequence(target)

    return predicted == target


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for MemNovo.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional path to log file
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )

    # Set MemNovo logger
    memnovo_logger = logging.getLogger('memnovo')
    memnovo_logger.setLevel(log_level)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device.

    Args:
        device: Device string ('cuda', 'cpu', or None for auto)

    Returns:
        torch.device instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }
