## Installation

Requirements:

* NVIDIA GPU with a driver that supports CUDA 11.8
* Conda (Miniconda or Anaconda)

We strongly recommend following the installation instructions below to ensure proper environment setup.

```bash
git clone https://github.com/IONSliu/Bridgediff.git
cd Bridgediff
conda env create -f environment.yml
conda activate cloth
```

This creates a conda environment named `cloth` (Python 3.9, PyTorch 2.1.0 + CUDA 11.8). All Python packages are listed in `environment.yml`.

For IP-Adapter, the required pretrained weights `ip-adapter_sd15.bin` can be downloaded from:
https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter_sd15.bin
