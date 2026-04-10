"""
Data Handler

Utilities for loading and preprocessing mass spectrometry data.
"""

import numpy as np
import pandas as pd
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Unified data handler for mass spectrometry files.

    Supports:
    - MGF (Mascot Generic Format)
    - CSV/TSV with spectrum columns
    - Parquet files

    Args:
        config: Configuration dictionary with 'path', 'format', 'max_samples'
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.path = config.get('path', '')
        self.format = config.get('format', 'auto')
        self.max_samples = config.get('max_samples', -1)
        self.seed = config.get('seed', 42)

    def load_data(self) -> pd.DataFrame:
        """
        Load data from file.

        Returns:
            DataFrame with spectrum data
        """
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        # Auto-detect format
        if self.format == 'auto':
            self.format = self._detect_format(path)

        logger.info(f"Loading {self.format} file: {path}")

        # Load based on format
        if self.format == 'mgf':
            df = self._load_mgf(path)
        elif self.format == 'csv':
            df = pd.read_csv(path)
        elif self.format == 'tsv':
            df = pd.read_csv(path, sep='\t')
        elif self.format == 'parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        logger.info(f"Loaded {len(df)} spectra")

        # Sample if needed
        if 0 < self.max_samples < len(df):
            df = df.sample(n=self.max_samples, random_state=self.seed)
            logger.info(f"Sampled {len(df)} spectra")

        return df

    def _detect_format(self, path: Path) -> str:
        """Auto-detect file format from extension."""
        suffix = path.suffix.lower()
        format_map = {
            '.mgf': 'mgf',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.parquet': 'parquet',
            '.pq': 'parquet',
        }
        return format_map.get(suffix, 'csv')

    def _load_mgf(self, path: Path) -> pd.DataFrame:
        """Load MGF file."""
        return load_mgf_file(str(path))


def load_mgf_file(path: str) -> pd.DataFrame:
    """
    Load MGF (Mascot Generic Format) file.

    Args:
        path: Path to MGF file

    Returns:
        DataFrame with spectrum data
    """
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
                parts = line[8:].split()
                spectrum['precursor_mz'] = float(parts[0])
                if len(parts) > 1:
                    spectrum['precursor_intensity'] = float(parts[1])

            elif line.startswith("CHARGE="):
                charge_str = line[7:].strip()
                spectrum['precursor_charge'] = int(charge_str.rstrip('+-'))

            elif line.startswith("TITLE="):
                spectrum['spectrum_id'] = line[6:].strip()

            elif line.startswith("SEQ="):
                spectrum['sequence'] = line[4:].strip()

            elif line.startswith("RTINSECONDS="):
                spectrum['retention_time'] = float(line[12:])

            elif line.startswith("SCANS="):
                spectrum['scan'] = line[6:].strip()

            elif line.startswith("END IONS"):
                if mz_list:
                    spectrum['mz_array'] = np.array(mz_list, dtype=np.float32)
                    spectrum['intensity_array'] = np.array(intensity_list, dtype=np.float32)

                    # Set defaults
                    spectrum.setdefault('precursor_mz', 0.0)
                    spectrum.setdefault('precursor_charge', 2)
                    spectrum.setdefault('spectrum_id', f'spectrum_{len(spectra)}')
                    spectrum.setdefault('sequence', '')

                    spectra.append(spectrum)

            elif line and not line.startswith(('BEGIN', 'END', '#')):
                # Try to parse as m/z intensity pair
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz_list.append(float(parts[0]))
                        intensity_list.append(float(parts[1]))
                except (ValueError, IndexError):
                    pass

    logger.info(f"Loaded {len(spectra)} spectra from MGF file")
    return pd.DataFrame(spectra)


def save_predictions(
    predictions: List[Dict[str, Any]],
    output_path: str,
    format: str = 'csv',
) -> None:
    """
    Save predictions to file.

    Args:
        predictions: List of prediction dictionaries
        output_path: Output file path
        format: Output format ('csv', 'tsv', 'json')
    """
    serializable = []
    for item in predictions:
        row = dict(item)
        if 'beam_predictions' in row and not isinstance(row['beam_predictions'], str):
            row['beam_predictions'] = json.dumps(row['beam_predictions'], ensure_ascii=False)
        serializable.append(row)

    if format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
        return

    df = pd.DataFrame(serializable)

    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'tsv':
        df.to_csv(output_path, sep='\t', index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported output format: {format}")

    logger.info(f"Saved {len(predictions)} predictions to {output_path}")
