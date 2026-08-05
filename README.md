# MemNovo

**Look Back at the Spectrum for Balanced De Novo Peptide Sequencing from Mass Spectrometry**

MemNovo is a training-free, plug-and-play inference-time enhancement for Transformer-based de novo peptide sequencing. It retains a persistent memory of the encoded spectrum and injects a small retrieved spectral signal at the final decoder layer. This helps the decoder balance its peptide-prefix prior with evidence from the input spectrum.

Paper: [arXiv:2606.11868](https://arxiv.org/abs/2606.11868)

## Installation

```bash
git clone https://github.com/AIMS-Lab-HKUSTGZ/MemNovo.git
cd MemNovo
conda env create -f environment.yml
conda activate memnovo
pip install -e .
```

Alternatively, install the Python dependencies with:

```bash
pip install -r requirements.txt
pip install -e .
```

The repository includes the upstream source trees used by the experiment wrappers:

- `external/casanovo`
- `external/instanovo`
- `external/primenovo`

Download the Casanovo and InstaNovo checkpoints with:

```bash
bash scripts/download_models.sh
```

The paper configurations use Casanovo v5.0.0 and InstaNovo v1.1.0. PrimeNovo requires a separately obtained checkpoint at `weights/model_massive.ckpt`.

## Quick start

Run InstaNovo with MemNovo on an annotated MGF file:

```bash
python scripts/run_inference.py \
  --config configs/memnovo_instanovo.yaml \
  --input path/to/spectra.mgf \
  --output results/predictions.jsonl \
  --metrics-output results/metrics.json \
  --evaluate
```

For the matching baseline, use `configs/baseline_instanovo.yaml`. Casanovo configurations are available as `configs/memnovo_casanovo.yaml` and `configs/baseline_casanovo.yaml`.

## Nine Species benchmark: data provenance

The experiments use the **reprocessed Nine Species benchmark** associated with Casanovo, rather than the original DeepNovo V1 benchmark. This benchmark is often called the V2 version.

- Original benchmark: Tran et al., *De novo peptide sequencing by deep learning* (2017).
- Reprocessed benchmark used here: Wen and Noble, *A multi-species benchmark for training and validating mass spectrometry proteomics machine learning models* (2024), available from [Zenodo record 13685813](https://zenodo.org/records/13685813).

The reprocessed release provides both `main` and `balanced` variants. For paper reproduction, begin from the full reprocessed data used by the Casanovo workflow; record the downloaded archive, checksum, and any local conversion steps in your experiment log.

### Important note on reported spectrum counts

MemNovo runs the official Casanovo and InstaNovo code paths and preserves their respective data loading, preprocessing, and input-validity rules. Therefore, the number reported for a particular experiment is the number of spectra that **enter that model's inference/evaluation pipeline**, not necessarily the raw PSM count in an upstream data archive. Some records can be excluded by model-specific preprocessing or input constraints.

For reproducible comparisons, report all of the following with every run:

- Zenodo archive and checksum;
- upstream Casanovo/InstaNovo commit or release;
- MemNovo commit and configuration file;
- per-species input count and retained/inferred count; and
- metric settings, including I/L normalization and mass tolerance.

### Expected local layout

The supplied nine-species scripts expect the following paths relative to the repository's parent directory:

```text
dataset/
├── NS1/
│   ├── Saccharomyces-cerevisiae.mgf
│   └── Methanosarcina-mazei.mgf
├── NS2/
│   └── Bacillus-subtilis.mgf
└── NS3/
    ├── Apis-mellifera.mgf
    ├── Solanum-lycopersicum.mgf
    ├── Vigna-mungo.mgf
    ├── Candidatus-endoloripes.mgf
    ├── H.-sapiens.mgf
    └── Mus-musculus.mgf
```

`scripts/download_data.sh` prints a data-acquisition guide. Before running the benchmark, place or link the annotated MGF files in the layout above, or update the paths in `scripts/run_nine_species.sh` for your local layout.

## Reproducing the paper experiments

Run all four combinations (Casanovo/InstaNovo, baseline/MemNovo):

```bash
bash scripts/run_nine_species.sh
```

Results are written under `results/nine_species/`. To use a smaller smoke-test subset, set `MEMNOVO_MAX_SAMPLES` before launching the script:

```bash
MEMNOVO_MAX_SAMPLES=1000 bash scripts/run_nine_species.sh
```

The sensitivity-scaling diagnostic can be run with:

```bash
bash scripts/run_sensitivity.sh instanovo
bash scripts/run_sensitivity.sh casanovo
```

These experiments use local test datasets configured in the scripts; see [docs/experiments.md](docs/experiments.md) for the expected paths and commands.

## Default MemNovo settings

The paper configuration uses:

```yaml
memnovo:
  enabled: true
  residual_scale: 0.005
  apply_to_last_n_layers: 1
  target_layers: [-1]
  use_gating: false
```

The retrieval is applied only at the final decoder layer. Larger residual scales or injection into more layers can disrupt the pretrained model representation; see the paper for ablations.

## Citation

If you use MemNovo, please cite:

```bibtex
@inproceedings{lyu2026memnovo,
  title={MemNovo: Look Back at the Spectrum for Balanced De Novo Peptide Sequencing from Mass Spectrometry},
  author={Lyu, Dongxin and Zhou, Jingbo and Xiang, Hongxin and Li, Yuqiang and Xia, Jun},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

Please also cite the relevant upstream model and benchmark publications when using the supplied experimental setup.

## License

This project is released under the [MIT License](LICENSE).
