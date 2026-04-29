# OASIS: Optimizer-Aware Subspace Importance Selection

OASIS is a gradient compression engine designed to drastically reduce the communication and memory overhead of deep learning training without sacrificing convergence fidelity. It achieves this through **Adaptive Rank Subspace Tracking**, dynamically allocating compression budgets per-layer based on gradient energy and intrinsic dimension.

## Key Features

- **Adaptive Rank Selection**: Instead of hard-coding a fixed rank (like traditional SVD-based methods), OASIS tracks the gradient energy dynamically. Layers with low-rank gradients use fewer components, while layers with complex gradients get higher rank capacity, optimizing the total memory budget.
- **OASIS-Track**: The primary operating mode uses a warm-started subspace power iteration combined with a tiny core SVD. This eliminates the heavy O(N^3) cost of full SVD during backpropagation.
- **Optimizer-Aware Compression**: Integrates directly with stateful optimizers (like AdamW) by preserving the gradient components that matter most to the optimizer's momentum and variance estimates.
- **TPU/XLA Ready**: Includes `train_tpu.py` specifically optimized for Google Cloud TPU VMs utilizing `torch_xla` and `MpDeviceLoader` for peak throughput.

## Usage

### 1. Standard Training (GPU/CPU)

Run the OASIS-Track adaptive gradient compression on CIFAR-100:

```bash
python3 train_cifar10.py --mode track --r-max 72 --energy-tau 0.97 --epochs 20 --batch-size 256 --dataset cifar100
```

### 2. TPU Training (Google Cloud)

If you have a Google Cloud TPU node with `torch_xla` installed, you can leverage the highly parallelized XLA script:

```bash
python3 train_tpu.py --mode track --r-max 72 --energy-tau 0.97 --epochs 20 --batch-size 256 --dataset cifar100
```

### 3. Running the Ablation Suite

To reproduce the efficiency vs. fidelity metrics comparing Adaptive Rank against Fixed-Rank baselines:

```bash
bash run_ablations.sh
```

Once the suite finishes, run the parsing script to automatically generate the markdown table for the paper:

```bash
python3 parse_results.py
```

## Architecture Details

- **`oasis/compressor.py`**: The core engine. Contains the adaptive rank logic (`_track_compress`) and exact SVD fallbacks.
- **`oasis/logger.py`**: High-fidelity instrumentation. Accurately tracks compute-only time vs. wall-clock time by isolating the Python formatting and print overhead.
- **`train_cifar10.py` / `train_baseline.py`**: The training loops heavily instrumented with `torch.cuda.synchronize()` for honest overhead reporting.

## Results

Empirical results demonstrate that OASIS-Track (Adaptive) consistently outperforms fixed-rank baselines by achieving a higher compressed gradient ratio while maintaining identical or superior validation accuracy.
