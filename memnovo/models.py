"""
Runnable model wrappers for paper-final MemNovo inference.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .backends import (
    ensure_external_imports,
    load_casanovo_backend,
    load_instanovo_backend,
    load_primenovo_backend,
    resolve_path,
)
from .manager import MemNovoManager

logger = logging.getLogger(__name__)


class MemNovoModel:
    """High-level wrapper around official Casanovo/InstaNovo/PrimeNovo inference."""

    SUPPORTED_MODELS = ["instanovo", "casanovo", "primenovo"]

    def __init__(
        self,
        model_name: str,
        model: Any,
        runtime: Dict[str, Any],
        memnovo_config: Dict[str, Any],
        inference_config: Dict[str, Any],
    ):
        self.model_name = model_name
        self.model = model
        self.runtime = runtime
        self.inference_config = inference_config
        self.memnovo_manager = MemNovoManager(memnovo_config)

        if self.memnovo_manager.is_enabled:
            self.memnovo_manager.register(model)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        device: str = "cuda",
    ) -> "MemNovoModel":
        import yaml

        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{model_name}'. Supported: {cls.SUPPORTED_MODELS}")

        if config is None:
            cfg = cls._get_default_config(model_name)
        elif isinstance(config, str):
            with open(config, "r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
        else:
            cfg = dict(config)

        model_cfg = dict(cfg.get("model", {}))
        memnovo_cfg = dict(cfg.get("memnovo", {}))
        inference_cfg = dict(cfg.get("inference", {}))

        model_cfg.setdefault("name", model_name)
        if checkpoint_path is None:
            checkpoint_path = model_cfg.get("checkpoint", model_cfg.get("model_path"))
        if checkpoint_path is None:
            checkpoint_path = cls._get_default_checkpoint(model_name)

        if model_name == "instanovo":
            model, runtime = load_instanovo_backend(checkpoint_path, device, inference_cfg)
        elif model_name == "primenovo":
            model, runtime = load_primenovo_backend(checkpoint_path, device, cfg)
        else:
            model, runtime = load_casanovo_backend(checkpoint_path, device, inference_cfg)

        return cls(model_name, model, runtime, memnovo_cfg, inference_cfg)

    @classmethod
    def _get_default_config(cls, model_name: str) -> Dict[str, Any]:
        return {
            "model": {
                "name": model_name,
                "checkpoint": cls._get_default_checkpoint(model_name),
            },
            "memnovo": {
                "enabled": True,
                "residual_scale": 0.005,
                "apply_to_last_n_layers": 1,
                "confidence_threshold": None,
                "use_softmax": True,
            },
            "inference": {
                "beam_size": 5,
                "batch_size": 64,
                "max_length": 30 if model_name == "instanovo" else (40 if model_name == "primenovo" else 100),
                "use_knapsack": model_name == "instanovo",
                "fp16": False,
                "save_beams": model_name == "instanovo",
            },
        }

    @classmethod
    def _get_default_checkpoint(cls, model_name: str) -> str:
        defaults = {
            "instanovo": "../weights/instanovo-v1.1.0.ckpt",
            "casanovo": "../weights/casanovo_v5_0_0.ckpt",
            "primenovo": "../weights/model_massive.ckpt",
        }
        return defaults[model_name]

    def predict(
        self,
        spectra: Union[str, List[Dict[str, Any]]],
        batch_size: Optional[int] = None,
        beam_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if isinstance(spectra, str):
            spectra = self._load_spectra(spectra)

        effective_batch_size = int(batch_size or self.inference_config.get("batch_size", 64))
        effective_beam_size = int(beam_size or self.inference_config.get("beam_size", 5))

        if self.model_name == "instanovo":
            return self._predict_instanovo(
                spectra,
                batch_size=effective_batch_size,
                beam_size=effective_beam_size,
                max_length=int(kwargs.get("max_length", self.inference_config.get("max_length", 30))),
                fp16=bool(kwargs.get("fp16", self.inference_config.get("fp16", True))),
            )

        if self.model_name == "primenovo":
            return self._predict_primenovo(
                spectra,
                batch_size=effective_batch_size,
                beam_size=effective_beam_size,
            )

        return self._predict_casanovo(
            spectra,
            batch_size=effective_batch_size,
            beam_size=effective_beam_size,
        )

    def _predict_instanovo(
        self,
        spectra_data: List[Dict[str, Any]],
        batch_size: int,
        beam_size: int,
        max_length: int,
        fp16: bool,
    ) -> List[Dict[str, Any]]:
        import polars as pl
        import torch
        from torch.utils.data import DataLoader

        ensure_external_imports()
        from instanovo.transformer.dataset import SpectrumDataset, collate_batch
        from instanovo.utils import SpectrumDataFrame

        if not spectra_data:
            return []

        decoder = self.runtime["decoder"]
        device = self.runtime["device"]
        model_config = self.runtime["model_config"] or {}
        max_charge = int(model_config.get("max_charge", 10))
        mass_tolerance = float(self.inference_config.get("precursor_mass_tolerance", 50)) * 1e-6

        frame = pl.DataFrame(
            {
                "experiment_name": [f"s{i}" for i in range(len(spectra_data))],
                "index": list(range(len(spectra_data))),
                "sequence": [item.get("sequence", "") for item in spectra_data],
                "modified_sequence": [item.get("modified_sequence", "") for item in spectra_data],
                "precursor_mz": [float(item.get("precursor_mz", 0.0)) for item in spectra_data],
                "precursor_charge": [int(item.get("precursor_charge", 0)) for item in spectra_data],
                "mz_array": [self._to_list(item.get("mz_array", [])) for item in spectra_data],
                "intensity_array": [self._to_list(item.get("intensity_array", [])) for item in spectra_data],
            }
        )

        sdf = SpectrumDataFrame(frame)
        original_count = len(sdf)
        sdf.filter_rows(lambda row: 0 < row["precursor_charge"] <= max_charge)
        valid_indices = sdf.df["index"].to_list()

        if len(sdf) < original_count:
            logger.info("Filtered %s spectra with invalid precursor charge", original_count - len(sdf))

        dataset = SpectrumDataset(
            df=sdf,
            residue_set=self.runtime["residue_set"],
            n_peaks=200,
            return_str=True,
            annotated=False,
        )
        num_workers = int(self.inference_config.get("num_workers", 0))
        pin_memory = bool(self.inference_config.get("pin_memory", False))
        prefetch_factor = self.inference_config.get("prefetch_factor")
        dataloader_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "collate_fn": collate_batch,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
        dataloader = DataLoader(
            **dataloader_kwargs,
        )

        final_results: List[Optional[Dict[str, Any]]] = [None] * len(spectra_data)
        autocast_enabled = fp16 and device.type == "cuda"
        return_beam = bool(self.inference_config.get("save_beams", False))
        total_batches = len(dataloader)
        log_interval = int(self.inference_config.get("log_interval", 50))
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            spectra, precursors, spectra_mask, peptides, _ = batch
            spectra = spectra.to(device)
            precursors = precursors.to(device)

            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                scored_sequences = decoder.decode(
                    spectra=spectra,
                    precursors=precursors,
                    beam_size=beam_size,
                    max_length=max_length,
                    mass_tolerance=mass_tolerance,
                    max_isotope=1,
                    return_beam=return_beam,
                )

            start_index = batch_idx * batch_size
            for offset, spectrum_beams in enumerate(scored_sequences):
                valid_index = start_index + offset
                if valid_index >= len(valid_indices):
                    continue

                original_index = int(valid_indices[valid_index])
                beam_predictions = []

                if isinstance(spectrum_beams, list):
                    iterator = spectrum_beams
                else:
                    iterator = [spectrum_beams]

                for scored_seq in iterator:
                    if scored_seq and getattr(scored_seq, "sequence", None):
                        sequence = "".join(scored_seq.sequence)
                        score = float(scored_seq.sequence_log_probability)
                    else:
                        sequence = ""
                        score = 0.0
                    beam_predictions.append({"pred_peptide": sequence, "confidence": score})

                first = beam_predictions[0] if beam_predictions else {"pred_peptide": "", "confidence": 0.0}
                final_results[original_index] = {
                    "spectrum_id": spectra_data[original_index].get("spectrum_id", f"spectrum_{original_index}"),
                    "sequence": first["pred_peptide"],
                    "score": first["confidence"],
                    "beam_predictions": beam_predictions,
                    "num_beams": len(beam_predictions),
                    "precursor_mz": float(spectra_data[original_index].get("precursor_mz", 0.0)),
                    "precursor_charge": int(spectra_data[original_index].get("precursor_charge", 0)),
                    "true_sequence": spectra_data[original_index].get("sequence", ""),
                    "model": "instanovo_memnovo" if self.memnovo_manager.is_enabled else "instanovo",
                }

            self.memnovo_manager.reset()
            if log_interval > 0 and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches):
                elapsed = time.time() - start_time
                per_batch = elapsed / (batch_idx + 1)
                remaining = per_batch * (total_batches - batch_idx - 1)
                logger.info(
                    "InstaNovo batches %s/%s, elapsed %.1fs, eta %.1fs, %.3fs/batch",
                    batch_idx + 1,
                    total_batches,
                    elapsed,
                    remaining,
                    per_batch,
                )

        for index, result in enumerate(final_results):
            if result is None:
                final_results[index] = {
                    "spectrum_id": spectra_data[index].get("spectrum_id", f"spectrum_{index}"),
                    "sequence": "",
                    "score": 0.0,
                    "beam_predictions": [],
                    "num_beams": 0,
                    "precursor_mz": float(spectra_data[index].get("precursor_mz", 0.0)),
                    "precursor_charge": int(spectra_data[index].get("precursor_charge", 0)),
                    "true_sequence": spectra_data[index].get("sequence", ""),
                    "model": "instanovo_filtered",
                }

        return [result for result in final_results if result is not None]

    def _predict_casanovo(
        self,
        spectra_data: List[Dict[str, Any]],
        batch_size: int,
        beam_size: int,
    ) -> List[Dict[str, Any]]:
        import torch

        if not spectra_data:
            return []

        self.model.n_beams = beam_size
        self.model.top_match = beam_size
        device = self.runtime["device"]
        total_batches = (len(spectra_data) + batch_size - 1) // batch_size
        log_interval = int(self.inference_config.get("log_interval", 50))
        start_time = time.time()

        results: List[Dict[str, Any]] = []
        for batch_idx, start in enumerate(range(0, len(spectra_data), batch_size)):
            batch = spectra_data[start : start + batch_size]
            processed_batch = []
            processed_indices = []
            batch_results: List[Optional[Dict[str, Any]]] = [None] * len(batch)

            for idx, spectrum in enumerate(batch):
                processed = self._preprocess_casanovo_spectrum(spectrum)
                if processed is None:
                    batch_results[idx] = self._empty_result(spectrum, "casanovo_filtered")
                    continue
                processed_batch.append(processed)
                processed_indices.append(idx)

            if not processed_batch:
                results.extend([item for item in batch_results if item is not None])
                continue

            mz_arrays, intensity_arrays, precursor_mzs, precursor_charges = self._prepare_casanovo_batch(processed_batch)
            mz_arrays = mz_arrays.to(device)
            intensity_arrays = intensity_arrays.to(device)
            precursor_mzs = precursor_mzs.to(device)
            precursor_charges = precursor_charges.to(device)

            batch_dict = {
                "mz_array": mz_arrays,
                "intensity_array": intensity_arrays,
                "precursor_mz": precursor_mzs,
                "precursor_charge": precursor_charges,
            }

            try:
                with torch.no_grad():
                    predictions = self.model.forward(batch_dict)
            finally:
                self.memnovo_manager.reset()

            for processed_idx, prediction_list in enumerate(predictions):
                original_idx = processed_indices[processed_idx]
                spectrum = batch[original_idx]

                beam_predictions = []
                if prediction_list:
                    for pred in prediction_list:
                        score = float(pred[0]) if len(pred) > 0 else 0.0
                        sequence = str(pred[2]) if len(pred) > 2 else ""
                        beam_predictions.append({"pred_peptide": sequence, "confidence": score})

                first = beam_predictions[0] if beam_predictions else {"pred_peptide": "", "confidence": 0.0}
                batch_results[original_idx] = {
                    "spectrum_id": spectrum.get("spectrum_id", f"spectrum_{start + original_idx}"),
                    "sequence": first["pred_peptide"],
                    "score": first["confidence"],
                    "beam_predictions": beam_predictions,
                    "num_beams": len(beam_predictions),
                    "precursor_mz": float(spectrum.get("precursor_mz", 0.0)),
                    "precursor_charge": int(spectrum.get("precursor_charge", 0)),
                    "true_sequence": spectrum.get("sequence", ""),
                    "model": "casanovo_memnovo" if self.memnovo_manager.is_enabled else "casanovo",
                }

            results.extend([item for item in batch_results if item is not None])
            if log_interval > 0 and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches):
                elapsed = time.time() - start_time
                per_batch = elapsed / (batch_idx + 1)
                remaining = per_batch * (total_batches - batch_idx - 1)
                logger.info(
                    "Casanovo batches %s/%s, elapsed %.1fs, eta %.1fs, %.3fs/batch",
                    batch_idx + 1,
                    total_batches,
                    elapsed,
                    remaining,
                    per_batch,
                )

        return results

    def _predict_primenovo(
        self,
        spectra_data: List[Dict[str, Any]],
        batch_size: int,
        beam_size: int,
    ) -> List[Dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        if not spectra_data:
            return []

        self.model.n_beams = beam_size
        device = self.runtime["device"]
        total_batches = (len(spectra_data) + batch_size - 1) // batch_size
        log_interval = int(self.inference_config.get("log_interval", 50))
        start_time = time.time()
        results: List[Dict[str, Any]] = []

        for batch_idx, start in enumerate(range(0, len(spectra_data), batch_size)):
            batch = spectra_data[start : start + batch_size]
            processed_batch = []
            processed_indices = []
            batch_results: List[Optional[Dict[str, Any]]] = [None] * len(batch)

            for idx, spectrum in enumerate(batch):
                processed = self._preprocess_primenovo_spectrum(spectrum)
                if processed is None:
                    batch_results[idx] = self._empty_result(spectrum, "primenovo_filtered")
                    continue
                processed_batch.append(processed)
                processed_indices.append(idx)

            if not processed_batch:
                results.extend([item for item in batch_results if item is not None])
                continue

            spectra_tensor, precursors_tensor, labels = self._prepare_primenovo_batch(processed_batch)
            spectra_tensor = spectra_tensor.to(device)
            precursors_tensor = precursors_tensor.to(device)

            try:
                with torch.no_grad():
                    save_beams = bool(self.inference_config.get("save_beams", False))
                    if save_beams and hasattr(self.model, "ctc_decoder") and hasattr(self.model.ctc_decoder, "decoder"):
                        memory, memory_key_padding_mask = self.model.encoder(spectra_tensor)
                        output_logits, _, _ = self.model.decoder(
                            None,
                            precursors_tensor,
                            memory,
                            memory_key_padding_mask,
                        )
                        beam_results, beam_scores, _, out_lens = self.model.ctc_decoder.decoder.decode(
                            F.softmax(output_logits, dim=-1)
                        )
                        predicted_tokens = []
                        scores = []
                        beam_candidates: List[List[Dict[str, Any]]] = []
                        beam_limit = min(beam_size, beam_results.shape[1])
                        for sample_idx in range(beam_results.shape[0]):
                            sample_beams: List[Dict[str, Any]] = []
                            for beam_idx in range(beam_limit):
                                beam_len = int(out_lens[sample_idx, beam_idx].item())
                                token_ids = beam_results[sample_idx, beam_idx, :beam_len].tolist()
                                sequence = "".join(self.model.decoder.detokenize_truth(token_ids, True))
                                raw_score = float(beam_scores[sample_idx, beam_idx].item())
                                confidence = float((1.0 / torch.exp(beam_scores[sample_idx, beam_idx])).item())
                                sample_beams.append(
                                    {
                                        "pred_peptide": sequence,
                                        "confidence": confidence,
                                        "raw_score": raw_score,
                                    }
                                )
                            beam_candidates.append(sample_beams)
                            predicted_tokens.append(list(sample_beams[0]["pred_peptide"]) if sample_beams else [])
                            scores.append(sample_beams[0]["confidence"] if sample_beams else 0.0)
                    else:
                        beam_candidates = []
                        predicted_tokens, scores = self.model.forward(spectra_tensor, precursors_tensor, labels)
            finally:
                self.memnovo_manager.reset()

            scores_list = scores.detach().cpu().tolist() if hasattr(scores, "detach") else list(scores)
            for processed_idx, token_list in enumerate(predicted_tokens):
                original_idx = processed_indices[processed_idx]
                spectrum = batch[original_idx]
                sequence = "".join(token_list)
                score = float(scores_list[processed_idx]) if processed_idx < len(scores_list) else 0.0
                if save_beams and beam_candidates:
                    candidates = beam_candidates[processed_idx]
                    beam_predictions = [
                        {
                            "pred_peptide": cand["pred_peptide"],
                            "confidence": float(cand["confidence"]),
                        }
                        for cand in candidates
                    ]
                    if beam_predictions:
                        sequence = beam_predictions[0]["pred_peptide"]
                        score = float(beam_predictions[0]["confidence"])
                else:
                    beam_predictions = [{"pred_peptide": sequence, "confidence": score}]
                batch_results[original_idx] = {
                    "spectrum_id": spectrum.get("spectrum_id", f"spectrum_{start + original_idx}"),
                    "sequence": sequence,
                    "score": score,
                    "beam_predictions": beam_predictions,
                    "num_beams": len(beam_predictions),
                    "precursor_mz": float(spectrum.get("precursor_mz", 0.0)),
                    "precursor_charge": int(spectrum.get("precursor_charge", 0)),
                    "true_sequence": spectrum.get("sequence", ""),
                    "model": "primenovo_memnovo" if self.memnovo_manager.is_enabled else "primenovo",
                }

            results.extend([item for item in batch_results if item is not None])
            if log_interval > 0 and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches):
                elapsed = time.time() - start_time
                per_batch = elapsed / (batch_idx + 1)
                remaining = per_batch * (total_batches - batch_idx - 1)
                logger.info(
                    "PrimeNovo batches %s/%s, elapsed %.1fs, eta %.1fs, %.3fs/batch",
                    batch_idx + 1,
                    total_batches,
                    elapsed,
                    remaining,
                    per_batch,
                )

        return results

    def _preprocess_casanovo_spectrum(self, spectrum: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        mz = np.array(spectrum.get("mz_array", []), dtype=np.float32)
        intensity = np.array(spectrum.get("intensity_array", []), dtype=np.float32)
        precursor_mz = float(spectrum.get("precursor_mz", 0.0))
        precursor_charge = int(spectrum.get("precursor_charge", 0))

        if len(mz) == 0 or precursor_mz <= 0 or precursor_charge <= 0:
            return None

        max_charge = int(self.runtime.get("max_charge", 5))
        if precursor_charge > max_charge:
            return None

        mask = (mz >= 50.0) & (mz <= 2500.0)
        mz, intensity = mz[mask], intensity[mask]
        if len(mz) < 20:
            return None

        precursor_mask = np.abs(mz - precursor_mz) > 2.0
        mz, intensity = mz[precursor_mask], intensity[precursor_mask]
        if len(mz) < 20:
            return None

        if intensity.max(initial=0.0) > 0:
            intensity = intensity / intensity.max()
        intensity = np.sqrt(intensity)

        keep = intensity >= 0.01
        mz, intensity = mz[keep], intensity[keep]
        if len(mz) < 20:
            return None

        if len(mz) > 150:
            top_indices = np.argsort(intensity)[-150:]
            top_indices = np.sort(top_indices)
            mz = mz[top_indices]
            intensity = intensity[top_indices]

        norm = np.linalg.norm(intensity)
        if norm > 0:
            intensity = intensity / norm

        return {
            "mz_array": mz,
            "intensity_array": intensity,
            "precursor_mz": precursor_mz,
            "precursor_charge": precursor_charge,
            "spectrum_id": spectrum.get("spectrum_id"),
            "sequence": spectrum.get("sequence", ""),
        }

    def _prepare_casanovo_batch(self, batch: List[Dict[str, Any]]):
        import torch

        max_peaks = max(len(item["mz_array"]) for item in batch)
        batch_size = len(batch)

        mz_arrays = torch.zeros(batch_size, max_peaks, dtype=torch.float32)
        intensity_arrays = torch.zeros(batch_size, max_peaks, dtype=torch.float32)
        precursor_mzs = torch.zeros(1, batch_size, dtype=torch.float32)
        precursor_charges = torch.zeros(1, batch_size, dtype=torch.float32)

        for idx, spectrum in enumerate(batch):
            peak_count = len(spectrum["mz_array"])
            mz_arrays[idx, :peak_count] = torch.tensor(spectrum["mz_array"], dtype=torch.float32)
            intensity_arrays[idx, :peak_count] = torch.tensor(spectrum["intensity_array"], dtype=torch.float32)
            precursor_mzs[0, idx] = float(spectrum["precursor_mz"])
            precursor_charges[0, idx] = float(spectrum["precursor_charge"])

        return mz_arrays, intensity_arrays, precursor_mzs, precursor_charges

    def _preprocess_primenovo_spectrum(self, spectrum: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import spectrum_utils.spectrum as sus

        cfg = self.runtime.get("base_config", {})
        mz = np.array(spectrum.get("mz_array", []), dtype=np.float64)
        intensity = np.array(spectrum.get("intensity_array", []), dtype=np.float32)
        precursor_mz = float(spectrum.get("precursor_mz", 0.0))
        precursor_charge = int(spectrum.get("precursor_charge", 0))

        if len(mz) == 0 or precursor_mz <= 0 or precursor_charge <= 0:
            return None

        spectrum_obj = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz,
            intensity,
        )
        try:
            spectrum_obj.set_mz_range(float(cfg.get("min_mz", 1.0)), float(cfg.get("max_mz", 6500.0)))
            if len(spectrum_obj.mz) == 0:
                raise ValueError
            spectrum_obj.remove_precursor_peak(float(cfg.get("remove_precursor_tol", 1.0)), "Da")
            if len(spectrum_obj.mz) == 0:
                raise ValueError
            spectrum_obj.filter_intensity(float(cfg.get("min_intensity", 0.0)), int(cfg.get("n_peaks", 800)))
            if len(spectrum_obj.mz) == 0:
                raise ValueError
            spectrum_obj.scale_intensity("root", 1)
            intensities = spectrum_obj.intensity / np.linalg.norm(spectrum_obj.intensity)
        except ValueError:
            return None

        peaks = np.stack([spectrum_obj.mz.astype(np.float32), intensities.astype(np.float32)], axis=1)
        return {
            "spectra": peaks,
            "precursor_mz": precursor_mz,
            "precursor_charge": precursor_charge,
            "sequence": spectrum.get("sequence", spectrum.get("spectrum_id", "")),
            "spectrum_id": spectrum.get("spectrum_id", ""),
        }

    def _prepare_primenovo_batch(self, batch: List[Dict[str, Any]]):
        import torch

        max_peaks = max(item["spectra"].shape[0] for item in batch)
        batch_size = len(batch)
        spectra_tensor = torch.zeros(batch_size, max_peaks, 2, dtype=torch.float32)
        precursor_mzs = torch.tensor([float(item["precursor_mz"]) for item in batch], dtype=torch.float32)
        precursor_charges = torch.tensor([float(item["precursor_charge"]) for item in batch], dtype=torch.float32)
        labels = [item.get("sequence", item.get("spectrum_id", "")) for item in batch]

        for idx, item in enumerate(batch):
            peak_count = item["spectra"].shape[0]
            spectra_tensor[idx, :peak_count, :] = torch.tensor(item["spectra"], dtype=torch.float32)

        precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
        precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()
        return spectra_tensor, precursors, labels

    def _load_spectra(self, path: str) -> List[Dict[str, Any]]:
        from evaluation.data_handler import DataHandler

        handler = DataHandler({"path": path, "format": "auto"})
        df = handler.load_data()
        spectra: List[Dict[str, Any]] = []

        for index, row in df.iterrows():
            spectra.append(
                {
                    "spectrum_id": row.get("spectrum_id", f"spectrum_{index}"),
                    "mz_array": np.array(row["mz_array"], dtype=np.float32),
                    "intensity_array": np.array(row["intensity_array"], dtype=np.float32),
                    "precursor_mz": float(row.get("precursor_mz", 0.0)),
                    "precursor_charge": int(row.get("precursor_charge", 0)),
                    "sequence": row.get("sequence", ""),
                    "modified_sequence": row.get("modified_sequence", ""),
                }
            )

        return spectra

    def _empty_result(self, spectrum: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        return {
            "spectrum_id": spectrum.get("spectrum_id", ""),
            "sequence": "",
            "score": 0.0,
            "beam_predictions": [],
            "num_beams": 0,
            "precursor_mz": float(spectrum.get("precursor_mz", 0.0)),
            "precursor_charge": int(spectrum.get("precursor_charge", 0)),
            "true_sequence": spectrum.get("sequence", ""),
            "model": model_name,
        }

    def cleanup(self) -> None:
        self.memnovo_manager.unregister()
        self.runtime.clear()
        self.model = None

    def get_stats(self) -> Dict[str, Any]:
        return self.memnovo_manager.get_stats()

    @property
    def device(self):
        return self.runtime.get("device")

    @staticmethod
    def _to_list(values: Any) -> List[float]:
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    def __repr__(self) -> str:
        return f"MemNovoModel(model={self.model_name}, memnovo_enabled={self.memnovo_manager.is_enabled})"
