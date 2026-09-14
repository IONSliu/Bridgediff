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

## Pretrained Weights

The pretrained weights of BridgeDiff for both **VITON-HD** and **DressCode** can be downloaded from:

https://huggingface.co/IONSLIU1/Bridgediff-weights

For IP-Adapter, the required pretrained weights `ip-adapter-plus_sd15.bin` can be downloaded from:

https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-plus_sd15.bin

## Dataset

The datasets used in our experiments can be obtained from their official repositories:

* VITON-HD: https://github.com/shadow2496/VITON-HD
* DressCode: https://github.com/aimagelab/dress-code

> **Note:** Due to upload size limitations, and because we cannot directly redistribute certain files from the official datasets, some additional files are not included in the current release. For example, the encoder-extracted features required during training and inference will be uploaded separately in a subsequent update. In addition, some data required for training and inference must be obtained directly from the official VITON-HD and DressCode datasets. Please download the corresponding datasets from their official repositories listed above.

## Training and Inference

Please ensure that the required datasets, pretrained weights, and corresponding configuration parameters are correctly prepared before running the training or inference scripts.

### Training

To train BridgeDiff, run:

```bash
bash run.sh
```

### Inference

To perform inference with BridgeDiff, run:

```bash
python stage2test.py
```

## Acknowledgements

Our code is developed based on the excellent [IMAGDressing](https://github.com/muzishen/IMAGDressing) project. We also gratefully acknowledge [IDM-VTON](https://github.com/yisol/IDM-VTON) for their valuable contributions to the virtual try-on community. We sincerely thank the authors of these projects for making their code and models publicly available, which greatly facilitated our research and the development of BridgeDiff.

