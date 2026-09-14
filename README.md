# BridgeDiff

## Installation

Requirements:

- NVIDIA GPU with a driver that supports CUDA 11.8
- Conda ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda)

```bash
git clone https://github.com/IONSliu/Bridgediff.git
cd Bridgediff
conda env create -f environment.yml
conda activate cloth
```

This creates a conda environment named `cloth` (Python 3.9, PyTorch 2.1.0 + CUDA 11.8). All Python packages are listed in `environment.yml`.
