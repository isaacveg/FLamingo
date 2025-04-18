# FLamingo

A comprehensive and flexible MPI-based library for Federated Learning research and development.

*Inspired by and incorporating elements from PFL-Non-IID and FedLab*

## Overview

FLamingo is a Python library designed for researchers and practitioners working in Federated Learning (FL). It provides a flexible framework built on MPI for efficient communication in distributed environments, allowing users to easily implement and experiment with various FL algorithms, client-server architectures, and data distributions.

## Project Structure

```
/
├── FLamingo/           # Core library (client, server, network, utils)
├── datasets/           # Dataset generation scripts and utilities
│   └── leaf_data/      # Special handling for LEAF benchmark datasets
├── models/             # Common ML model implementations
├── templates/          # Templates for creating new FL methods
├── analysis_tools/     # Tools for analyzing experiment results
├── config/             # Configuration files
└── docs/               # Documentation
```

## Features

- **MPI-based Communication**: Efficient message passing between server and clients
- **Flexible Architecture**: Customize clients, servers, models, and dataset partitioning
- **Rich Model Zoo**: Ready-to-use implementations of common deep learning models
- **Dataset Management**: Tools to generate and partition standard FL benchmark datasets
- **Extensible Templates**: Easily create your own FL algorithms with provided templates
- **Analysis Utilities**: Visualize and analyze experiment results

## Installation

### Option 1: Clone and Use Directly

```bash
git clone https://github.com/isaacveg/FLamingo.git
```

Then add the FLamingo directory to your Python path:

```python
# In your Python file
import sys
sys.path.append('/path/to/FLamingo')  # Replace with your actual path
```

### Option 2: Install as a Package

```bash
git clone https://github.com/isaacveg/FLamingo.git
cd FLamingo
pip install -e .  # Install in development mode
# OR
pip install .     # Standard installation
```

## Getting Started

### 1. Generate Dataset Partitions

FLamingo supports creating partitioned datasets for FL experiments in various distributions (IID, non-IID).

#### Using FLamingo_datasets (Recommended)

```bash
git clone https://github.com/isaacveg/FLamingo_datasets.git
cd FLamingo_datasets

# Example: Generate CIFAR-10 with 30 clients using IID distribution
python gen_cifar10.py --nc 30 --dist iid --seed 2024 --indir ../datasets/ --outdir ../datasets/

# For non-IID Dirichlet distribution
python gen_cifar10.py --nc 30 --dist dir --alpha 0.1 --seed 2024 --indir ../datasets/ --outdir ../datasets/
```

The generated dataset will be stored in `datasets/cifar10/` with:
- `train/` and `test/` directories containing `.npz` files for each client
- `config.json` with metadata
- `0.npz` containing the full dataset (for convenience)

#### LEAF Datasets

For LEAF benchmark datasets (FEMNIST, Shakespeare, etc.):

1. Download and preprocess the dataset following the official LEAF guidance
2. Use the script in FLamingo to convert to the required format:
   ```bash
   cd datasets/leaf_data
   ./gen_pickle_dataset.sh
   ```
3. Then use the appropriate generator script (e.g., `gen_femnist.py` or `gen_shakespeare.py`)

### 2. Running Experiments

#### Option 1: Using a Configuration File (Recommended)

Create a `config.yaml` file with your experiment parameters and use a main script to manage the process:

```bash
python main.py
```

#### Option 2: Using MPI Directly

```bash
mpiexec --oversubscribe -n 1 python path/to/server.py : -n 30 python path/to/client.py > run.log 2>&1
```

### 3. Example Projects

Check out [FLamingo_examples](https://github.com/isaacveg/FLamingo_examples) for ready-to-use example implementations.

## Creating Your Own FL Methods

FLamingo makes it easy to implement custom FL algorithms:

1. Copy a template from the `templates/` directory and rename it to your project name.
   - For example, copy `templates/template/` to `my_fl_method/`.

```bash
cp -r templates/template/ my_fl_method/
```

2. This creates a directory with template files:
   - `server.py`: Implement a `Server` class with a `run()` method
   - `client.py`: Implement a `Client` class with a `run()` method
   - `network.py`: Configure communication protocols
   - `main.py`: Entry point for your experiment
   - `config.yaml`: Configuration parameters

## Models

The `models/` directory contains implementations of common deep learning architectures:
- AlexNet
- VGG (various depths)
- ResNet (various depths)
- CNN
- DNN
- RNN
- MobileNet
- Linear Regression
- Encoder-Decoder architectures

Import and modify these models as needed for your experiments.

## Hyperparameter Reference

Please refer to `docs/hyperparameters.md` for a detailed list of tested hyperparameters used in FLamingo.

## Contribution

Contributions are welcome! Please feel free to:
- Submit issues for bugs or feature requests
- Create pull requests with improvements
- Share your implementation of FL algorithms using FLamingo

## License

FLamingo is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use FLamingo in your research, please cite:

```
@misc{FLamingo2023,
    author = {Isaac Zhu and contributors},
    title = {FLamingo: A Flexible MPI-based Library for Federated Learning},
    year = {2023},
    howpublished = {\url{https://github.com/isaacveg/FLamingo}},
    note = {Accessed: [Insert date of access]}
}
```

## Acknowledgments

FLamingo incorporates and builds upon code from:
- [PFL-Non-IID](https://github.com/Tsingz0/PFL-Non-IID)
- [FedLab](https://github.com/SMILELab-FL/FedLab)

We also thank the LEAF project for benchmark datasets.
