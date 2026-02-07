"""
MemNovo Model Wrappers

Provides high-level interfaces for using MemNovo with Casanovo and InstaNovo models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
import yaml

from .manager import MemNovoManager

logger = logging.getLogger(__name__)


class MemNovoModel:
    """
    High-level wrapper for MemNovo-enhanced de novo sequencing.

    This class provides a unified interface for:
    1. Loading pre-trained models (Casanovo or InstaNovo)
    2. Applying MemNovo enhancement
    3. Running inference with beam search

    Example:
        >>> model = MemNovoModel.from_pretrained("instanovo", config="configs/memnovo.yaml")
        >>> predictions = model.predict(spectra)
    """

    SUPPORTED_MODELS = ['instanovo', 'casanovo']

    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        memnovo_config: Dict[str, Any],
        inference_config: Dict[str, Any],
    ):
        """
        Initialize MemNovo model wrapper.

        Args:
            model_name: Name of the base model ('instanovo' or 'casanovo')
            model: The loaded base model
            memnovo_config: MemNovo configuration
            inference_config: Inference parameters (beam_size, batch_size, etc.)
        """
        self.model_name = model_name
        self.model = model
        self.inference_config = inference_config

        # Initialize MemNovo manager
        self.memnovo_manager = MemNovoManager(memnovo_config)

        # Register hooks
        if memnovo_config.get('enabled', True):
            self.memnovo_manager.register(model)

        logger.info(f"MemNovoModel initialized with {model_name}")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        device: str = 'cuda',
    ) -> 'MemNovoModel':
        """
        Load a pre-trained model with MemNovo enhancement.

        Args:
            model_name: 'instanovo' or 'casanovo'
            checkpoint_path: Path to model checkpoint (optional, uses default if not provided)
            config: Path to YAML config or config dictionary
            device: Target device ('cuda' or 'cpu')

        Returns:
            MemNovoModel instance
        """
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported: {cls.SUPPORTED_MODELS}"
            )

        # Load configuration
        if config is None:
            cfg = cls._get_default_config(model_name)
        elif isinstance(config, str):
            with open(config, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = config

        # Get model-specific config
        model_cfg = cfg.get('model', {})
        memnovo_cfg = cfg.get('memnovo', cfg.get('crossatt_memvr_config', {}))
        inference_cfg = cfg.get('inference', {})

        # Set checkpoint path
        if checkpoint_path is None:
            checkpoint_path = model_cfg.get('checkpoint', model_cfg.get('model_path'))

        if checkpoint_path is None:
            checkpoint_path = cls._get_default_checkpoint(model_name)

        # Load base model
        model = cls._load_base_model(model_name, checkpoint_path, model_cfg, device)

        # Create instance
        return cls(model_name, model, memnovo_cfg, inference_cfg)

    @classmethod
    def _get_default_config(cls, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model."""
        return {
            'model': {
                'name': model_name,
            },
            'memnovo': {
                'enabled': True,
                'residual_scale': 0.005,
                'apply_to_last_n_layers': 1,
            },
            'inference': {
                'beam_size': 5,
                'batch_size': 64,
                'max_length': 40,
            },
        }

    @classmethod
    def _get_default_checkpoint(cls, model_name: str) -> str:
        """Get default checkpoint path for a model."""
        paths = {
            'instanovo': 'weights/instanovo-v1.1.0.ckpt',
            'casanovo': 'weights/casanovo_v5_0_0.ckpt',
        }
        return paths.get(model_name, '')

    @classmethod
    def _load_base_model(
        cls,
        model_name: str,
        checkpoint_path: str,
        model_cfg: Dict[str, Any],
        device: str,
    ) -> nn.Module:
        """Load the base de novo sequencing model."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Please download it first using 'bash scripts/download_models.sh'"
            )

        logger.info(f"Loading {model_name} from {checkpoint_path}")

        if model_name == 'instanovo':
            return cls._load_instanovo(checkpoint_path, model_cfg, device)
        elif model_name == 'casanovo':
            return cls._load_casanovo(checkpoint_path, model_cfg, device)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @classmethod
    def _load_instanovo(
        cls,
        checkpoint_path: str,
        model_cfg: Dict[str, Any],
        device: str,
    ) -> nn.Module:
        """Load InstaNovo model."""
        try:
            from instanovo.transformer.model import InstaNovo
            from instanovo.utils.residues import ResidueSet

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Get model config
            if 'config' in checkpoint:
                cfg = checkpoint['config']
            else:
                cfg = model_cfg

            # Initialize residue set
            residue_set = ResidueSet.from_config(cfg.get('residues', {}))

            # Create model
            model = InstaNovo(
                residue_set=residue_set,
                **cfg.get('model', {}),
            )

            # Load weights
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            model = model.to(device)
            model.eval()

            logger.info("InstaNovo loaded successfully")
            return model

        except ImportError:
            raise ImportError(
                "InstaNovo not found. Please install it: "
                "pip install instanovo or use git submodule"
            )

    @classmethod
    def _load_casanovo(
        cls,
        checkpoint_path: str,
        model_cfg: Dict[str, Any],
        device: str,
    ) -> nn.Module:
        """Load Casanovo model."""
        try:
            from casanovo.denovo.model import Spec2Pep

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Get model config
            if 'config' in checkpoint:
                cfg = checkpoint['config']
            else:
                cfg = model_cfg

            # Create model
            model = Spec2Pep(**cfg)

            # Load weights
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)

            model = model.to(device)
            model.eval()

            logger.info("Casanovo loaded successfully")
            return model

        except ImportError:
            raise ImportError(
                "Casanovo not found. Please install it: "
                "pip install casanovo or use git submodule"
            )

    def predict(
        self,
        spectra: Union[str, List[Dict[str, Any]]],
        batch_size: Optional[int] = None,
        beam_size: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Run de novo sequencing on input spectra.

        Args:
            spectra: Path to MGF file or list of spectrum dictionaries
            batch_size: Override batch size
            beam_size: Override beam size
            **kwargs: Additional inference parameters

        Returns:
            Dictionary mapping spectrum IDs to predicted peptide sequences
        """
        # Load spectra if path provided
        if isinstance(spectra, str):
            spectra = self._load_spectra(spectra)

        # Get inference parameters
        batch_sz = batch_size or self.inference_config.get('batch_size', 64)
        beam_sz = beam_size or self.inference_config.get('beam_size', 5)

        logger.info(f"Running inference on {len(spectra)} spectra")

        # Run inference
        predictions = {}
        with torch.no_grad():
            for i in range(0, len(spectra), batch_sz):
                batch = spectra[i:i + batch_sz]
                batch_preds = self._predict_batch(batch, beam_sz, **kwargs)
                predictions.update(batch_preds)

        return predictions

    def _predict_batch(
        self,
        batch: List[Dict[str, Any]],
        beam_size: int,
        **kwargs,
    ) -> Dict[str, str]:
        """Run prediction on a single batch."""
        # This is a placeholder - actual implementation depends on the model
        # For now, return empty predictions
        logger.warning("Batch prediction not fully implemented - using model directly")
        return {}

    def _load_spectra(self, path: str) -> List[Dict[str, Any]]:
        """Load spectra from file."""
        from evaluation.data_handler import DataHandler

        handler = DataHandler({'path': path, 'format': 'mgf'})
        df = handler.load_data()

        spectra = []
        for _, row in df.iterrows():
            spectra.append({
                'spectrum_id': row.get('spectrum_id', ''),
                'mz_array': row.get('mz_array'),
                'intensity_array': row.get('intensity_array'),
                'precursor_mz': row.get('precursor_mz'),
                'precursor_charge': row.get('precursor_charge'),
            })

        return spectra

    def enable_memnovo(self) -> None:
        """Enable MemNovo enhancement."""
        if not self.memnovo_manager.is_registered:
            self.memnovo_manager.register(self.model)

    def disable_memnovo(self) -> None:
        """Disable MemNovo enhancement (baseline mode)."""
        self.memnovo_manager.unregister()

    def get_stats(self) -> Dict[str, Any]:
        """Get MemNovo statistics."""
        return self.memnovo_manager.get_stats()

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device

    def __repr__(self) -> str:
        return (
            f"MemNovoModel("
            f"model={self.model_name}, "
            f"memnovo_enabled={self.memnovo_manager.is_enabled}, "
            f"device={self.device})"
        )
