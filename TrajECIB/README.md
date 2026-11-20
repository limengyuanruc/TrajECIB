# TrajECIB

Official PyTorch implementation of **TrajECIB**: an environment-aware causal
trajectory classification framework. TrajECIB separates **causal
(environment-invariant)** and **spurious (environment-dependent)** trajectory
features through a dual-stream encoder, an environment codebook, and an
information-bottleneck (IB) point gate, and is evaluated under noise,
class-imbalance, few-shot, region-shift, and cross-city settings.

## Repository Structure

```
TrajECIB/
├── main.py                       # Entry point: training / testing / robustness / explainability
├── args.py                       # All command-line arguments and hyperparameters
├── model/
│   ├── trajcausal_dualstream.py  # TrajECIB model (dual-stream) + reconstruction decoder
│   ├── trajcausal_baseline.py    # Baseline without the causal module (model_name contains "base")
│   ├── my_base_block.py          # EnvCodebook, prompt decoder, PointAttention (IB gate), ODL loss
│   ├── enc_block.py              # Unified sequence-encoder wrapper (GRU / LSTM / GRU-D / STGN / TrajFormer)
│   ├── grud.py                   # GRU-D and STGN cells
│   ├── TrajFormer.py             # Transformer trajectory encoder
│   └── base_block.py             # Readout (classification) heads
├── trainer/
│   ├── causal_trainer.py         # Training, evaluation, robustness and explanation tests
│   └── plots.py                  # Attribution baselines (via Captum)
├── utils/
│   ├── helper.py                 # Dataloaders, seeding, weight init, result aggregation
│   └── logger.py                 # Logging utilities
└── base/
    ├── model.py                  # BaseModel (parameter counting, FLOPs)
    └── trainer.py                # BaseTrainer (optimizer, scheduler, checkpointing)
```

## Requirements

- Python 3.8+
- PyTorch (tested with CUDA GPUs; `--gpu -1` falls back to CPU)

```bash
pip install -r requirements.txt
```

## Data Preparation

The code reads **processed trajectory pickles** from
`{my_data_dir}/trajdata/<city>/processed_data/`:

| Dataset (`--model_name` tag) | Path |
|---|---|
| GeoLife / Beijing (`bj`) | `trajdata/geolife/processed_data/dataset.pkl` |
| Jakarta (`ja`) | `trajdata/jakarta/processed_data/dataset.pkl` |
| Singapore (`sa`) | `trajdata/singapore/processed_data/dataset.pkl` |
| SHL (`shl`) | `trajdata/shl/processed_data/shl_dataset.pkl` |
| Weihai (`wh`) | `trajdata/weihai/processed_data/wh_dataset.pkl` |
| Yantai (`yt`) | `trajdata/yantai/processed_data/yt_dataset.pkl` |

Each pickle contains the dataset splits consumed by `utils/helper.py`
(`get_dataloader_dict`). Every sample consists of a variable-length
trajectory feature sequence (coordinates, elapsed time, distance, velocity
and acceleration, followed by 24 environmental context features), its
normalized version, the class label, and an attention mask. Public data
sources (GeoLife, Grab-Posisi, SHL) can be obtained from their official
websites; due to license terms we do not redistribute the data.

Region-shift robustness additionally uses `dataset_region1.pkl` /
`dataset_region2.pkl` in the same directory (see `main.py`).

## Usage

All commands are executed from the repository root, with `--my_data_dir ./`
pointing at this folder (data under `trajdata/`, logs written to `logs/`).

**Training (Beijing / GeoLife, GRU backbone):**

```bash
python main.py --my_data_dir ./ --mode train --model_name 2-bj \
  --base_encoder gru --envbook_num 25 --kl_loss 0.0001 --odl_loss 1 \
  --env_encoder 1 --max_epochs 200 --gpu 0 --num_driver 4 \
  --motion_dim 0 --c 0 --co 0 --local_att 0 --explain 0 \
  --robusttest 1 --roubust noise --name 261105_2-bj-gru
```

**Testing a trained checkpoint:**

```bash
python main.py --my_data_dir ./ --mode test --model_name 2-bj \
  --base_encoder gru --envbook_num 25 --kl_loss 0.0001 --odl_loss 1 \
  --env_encoder 1 --max_epochs 200 --gpu 0 --num_driver 4 \
  --motion_dim 0 --c 0 --co 0 --local_att 0 --explain 0 \
  --robusttest 1 --roubust noise --name 261105_2-bj-gru
```

Notes:

- Training runs over a fixed list of 10 random seeds and reports mean ± std
  of validation/test accuracy into `logs/<date>/<city>/<name>-<base_encoder>/`.
- In `--mode test`, a previously trained checkpoint is located under
  `{my_data_dir}/trained_new/<city>/<all|imb|few>/` in a folder ending with
  `_<last-segment-of-name>-<base_encoder>` (see `main.py` for details), then
  copied into the log directory and evaluated.
- `--model_name` selects both the dataset (`bj`/`ja`/`sa`/`shl`/`wh`/`yt`) and
  the model variant (containing `base` → `trajcausal_baseline`, otherwise the
  full dual-stream TrajECIB).
- `--base_encoder` ∈ {`gru`, `lstm`, `grud`, `stgn`, `former`}.
- Robustness protocols (`--roubust`): `noise`, `imbalance` (with
  `--roubust_size`), `few`, `regionshift`. Additional diagnostic tests are
  available via `--explain`, `--spurioustest`, `--spuriouswotr`, `--envprob`,
  `--intervene`, `--repinter`.

## Citation

If you find this code useful, please cite our paper.
