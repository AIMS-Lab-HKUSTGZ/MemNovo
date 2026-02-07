#!/usr/bin/env python3
"""
MemNovo Inference Script

Run de novo peptide sequencing with MemNovo enhancement.

Usage:
    python scripts/run_inference.py --config configs/memnovo.yaml --input data.mgf --output results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memnovo import MemNovoModel
from memnovo.utils import load_config, setup_logging, load_spectra
from evaluation import Evaluator, save_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MemNovo de novo peptide sequencing"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input spectrum file (MGF format)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='predictions.csv',
        help='Output predictions file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Override model checkpoint path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=None,
        help='Override beam size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Computation device (cuda or cpu)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate predictions (requires sequences in input file)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("MemNovo Inference")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Override config with command line arguments
    if args.checkpoint:
        config['model']['checkpoint'] = args.checkpoint
    if args.batch_size:
        config['inference']['batch_size'] = args.batch_size
    if args.beam_size:
        config['inference']['beam_size'] = args.beam_size

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Load model
    logger.info("Loading model...")
    try:
        model_name = config.get('model', {}).get('name', 'instanovo')
        checkpoint = config.get('model', {}).get('checkpoint')

        model = MemNovoModel.from_pretrained(
            model_name=model_name,
            checkpoint_path=checkpoint,
            config=config,
            device=args.device,
        )
        logger.info(f"Model loaded: {model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Make sure you have downloaded the model checkpoint:")
        logger.info("  bash scripts/download_models.sh")
        sys.exit(1)

    # Load spectra
    logger.info(f"Loading spectra from {input_path}")
    spectra = load_spectra(str(input_path))
    logger.info(f"Loaded {len(spectra)} spectra")

    # Run inference
    logger.info("Running inference...")
    predictions = model.predict(
        spectra,
        batch_size=config.get('inference', {}).get('batch_size', 64),
        beam_size=config.get('inference', {}).get('beam_size', 5),
    )

    # Convert to list format
    pred_list = [
        {
            'spectrum_id': spec.get('spectrum_id', f'spectrum_{i}'),
            'predicted_sequence': predictions.get(spec.get('spectrum_id', f'spectrum_{i}'), ''),
            'target_sequence': spec.get('sequence', ''),
        }
        for i, spec in enumerate(spectra)
    ]

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(pred_list, str(output_path), format='csv')
    logger.info(f"Saved predictions to {output_path}")

    # Evaluate if requested
    if args.evaluate:
        logger.info("Evaluating predictions...")
        evaluator = Evaluator(
            normalize_il=config.get('evaluation', {}).get('normalize_il', True)
        )

        pred_dicts = [{'sequence': p['predicted_sequence']} for p in pred_list]
        truth_dicts = [{'sequence': p['target_sequence']} for p in pred_list]

        metrics = evaluator.evaluate(pred_dicts, truth_dicts)

        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        logger.info(f"  AA Precision:  {metrics['aa_precision']:.4f}")
        logger.info(f"  AA Recall:     {metrics['aa_recall']:.4f}")
        logger.info(f"  Pep Precision: {metrics['pep_precision']:.4f}")
        logger.info(f"  Pep Recall:    {metrics['pep_recall']:.4f}")
        logger.info(f"  Samples:       {metrics['n_samples']}")

    # Print MemNovo stats
    stats = model.get_stats()
    logger.info("=" * 50)
    logger.info("MemNovo Statistics:")
    logger.info(f"  Effective calls:  {stats.get('effective_calls', 0)}")
    logger.info(f"  Total calls:      {stats.get('total_calls', 0)}")

    logger.info("Done!")


if __name__ == '__main__':
    main()
