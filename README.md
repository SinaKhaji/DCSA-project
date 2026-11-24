# DCSA Project — Distributed Deep Learning on a Multi-Core CPU

This project trains a convolutional neural network on MNIST and compares single-process training with PyTorch DistributedDataParallel (DDP), a manual parameter server, and a manual ring all-reduce. All workers run on one machine, using CPUs and the Gloo backend for distributed runs.

## Setup

Use Python **3.12** and Git. The steps below cover Windows PowerShell and Linux. An internet connection is needed to install dependencies and download MNIST on the first run.

```bash
git clone https://github.com/SinaKhaji/DCSA-project.git
cd DCSA-project
```

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the pinned dependencies and check that PyTorch has Gloo support.

```bash
python -m pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert torch.distributed.is_gloo_available()"
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in the remaining commands.

## Run the training benchmarks

Run each command from the **repository root**. The first training script to run downloads MNIST into `../data/MNIST`, a directory next to the repository. Later runs reuse it. Start one training job at a time when benchmarking.

| Configuration | Command | Combined result |
| --- | --- | --- |
| Single process | `python code/train_single.py` | `results/single.json` |
| DDP, 2 processes | `python code/train_ddp.py` with `WORLD_SIZE = 2` | `results/DDP_2.json` |
| DDP, 4 processes | `python code/train_ddp.py` with `WORLD_SIZE = 4` | `results/DDP_4.json` |
| Parameter server, 4 processes | `python code/train_ring_ps.py --method ps --world-size 4` | `results/ps_4.json` |
| Ring all-reduce, 4 processes | `python code/train_ring_ps.py --method ring --world-size 4` | `results/ring_4.json` |

Set `WORLD_SIZE` near the top of [code/train_ddp.py](code/train_ddp.py) before each DDP run. The default is `2`. For the four-process run, change the setting to `4`, save the file, and run the same command again. The manual script accepts `--world-size 2` or `--world-size 4`.

### Training settings

All five configurations use the same CNN, five epochs, seed `0`, SGD with a learning rate of `0.01` and momentum of `0.9`, and one PyTorch CPU computation thread per training process. The configured global batch size is `128`, giving per-process batch sizes of `128`, `64`, and `32` for one, two, and four processes, respectively.

### How gradients are synchronized

| Method | How it works |
| --- | --- |
| Single process | One process trains on the full training set and updates the model directly. |
| DDP | `DistributedSampler` assigns each process a shard. DDP averages gradients across processes during backpropagation, and each process applies the same optimizer update. |
| Parameter server | Rank 0 also trains on a shard. After backpropagation, it collects the other workers’ gradients, averages them with its own, and sends the result back. |
| Ring all-reduce | Each worker exchanges gradient chunks with its neighbors. Reduce-scatter sums the chunks, all-gather distributes the sums, and each worker divides by the process count. |

The manual methods synchronize the initial model parameters and average gradients after each backward pass, before the optimizer step.

## Results and plots

The JSON files record training and test loss and accuracy for every epoch, training time per epoch, and final test accuracy. DDP also writes per-rank files such as `DDP_4_0.json`. Use the combined `DDP_2.json` and `DDP_4.json` files for comparisons.

```bash
python code/plot_results.py
```

The plotting script uses the JSON files in `results/` and writes [accuracy versus epoch](results/plots/acc_vs_epochs.pdf) and [loss versus epoch](results/plots/loss_vs_epochs.pdf) to `results/plots/` in PNG and PDF formats. Each figure shows training and test curves. The script reads one combined JSON file per configuration from the selected folder.

### Repeat a benchmark

1. Run the five configurations on the same machine with the same software versions and settings. Record the CPU model, operating system, Python version, and PyTorch version.
2. After all five configurations finish, copy their combined JSON files into a separate folder such as `benchmarks/run-01/`, keeping the filenames in the table above. Each training command overwrites its existing result files. Save a copy of any earlier results you want to keep before running the benchmarks.
3. Repeat for `benchmarks/run-02/` and `benchmarks/run-03/`. Plot one saved repeat with the command below. To show an average across repeats, first average each metric at each epoch and save one combined JSON per configuration. Use the same archived runs for the report’s table and plots, and state whether each figure shows a single repeat or a mean.

```bash
python code/plot_results.py --results-dir benchmarks/run-01 --output-dir benchmarks/run-01/plots
```

### Compare performance

| Metric | How to read it |
| --- | --- |
| Average epoch time | `avg_epoch_time` is the mean training time per epoch in one run. It excludes test evaluation. In distributed runs, each epoch uses the slowest worker’s training time. |
| Final accuracy | `final_acc` is the last epoch’s test accuracy as a fraction. Multiply by 100 for a percentage. |
| Speedup | Divide the baseline’s typical epoch time by the configuration’s typical epoch time. For three repeats, use the median of the three `avg_epoch_time` values for each configuration. |
| Manual synchronization time | `avg_sync_time` is the mean of the per-epoch maximum across workers. It includes time waiting inside the synchronization routine and its tensor operations. The gradient flattening and copy-back steps are outside that timer. DDP has no separate communication timing in this code. |

For the results table, report the median and standard deviation of the run-level epoch averages, and report the mean final accuracy across repeats. Use `avg_epoch_time` for speedup because the scripts time different portions of the overall run in `total_time`.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`code/train_single.py`](code/train_single.py) | Single-process CPU baseline |
| [`code/train_ddp.py`](code/train_ddp.py) | PyTorch DDP training |
| [`code/train_ring_ps.py`](code/train_ring_ps.py) | Manual parameter-server and ring all-reduce training |
| [`code/plot_results.py`](code/plot_results.py) | Loss and accuracy plots |
| [`utils/utils.py`](utils/utils.py) | Shared CNN, MNIST loaders, training and evaluation helpers, and JSON output |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies |
| [`results/`](results/) | Benchmark JSON files and plots |

## Troubleshooting

- **DDP still starts two workers.** Edit `WORLD_SIZE` in `code/train_ddp.py` and save it before starting training.
- **The distributed port is already in use.** Finish or stop the earlier training job. DDP uses `127.0.0.1:29500` and the manual methods use `127.0.0.1:29501`.
- **Plotting reports a missing combined DDP file.** Copy `DDP_2.json` or `DDP_4.json` into the selected results folder along with any per-rank files for that configuration.
