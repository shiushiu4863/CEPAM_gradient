# Communication-Efficient and Privacy-Adaptable Mechanism for Federated Learning with gradient updates

## Intrdocution

In this work, we further study CEPAM, a joint mechanism that achieve privacy while compressing gradient updates in FL. CEPAM leverages RSUQ, building upon its capibility to convert quantization distortion into an additive noise term with adjustable variance, independent of the quantized data. Within CEPAM, clients and server can tailor the privacy mechanism. In addition, the inherent natrue of RSUQ as a vector quantizer aids in reducing the compression ratio compared to its scalar counterparts. This repository contains a baisc PyTorch implementation of CEPAM for both Gaussian (CEPAM-Gaussian) and Laplace (CEPAM-Laplace) mechanisms. Additionally, we provide basci implementation of all the baseline mechanisms mentioned in the paper. 

## Usage

This code has been tested on Python 3.13.5, PyTorch 2.7.1 and CUDA 11.5.

### Prerequisite

1. PyTorch=2.7.1: https://pytorch.org
2. scipy
3. tqdm
4. matplotlib
5. torchinfo
6. TensorboardX: https://github.com/lanpa/tensorboardX

### Installation

```md
# Create a conda environment (optional)
conda create -n cepam python=3.12
conda activate cepam

# Install PyTorch (with CUDA support)
pip install torch torchvision torchaudio

# Install other dependencies
pip install scipy tqdm matplotlib torchinfo tensorboardX


### Training

For CEPAM Gaussian mechanism:

```md
python main.py \
    --exp_name fl_cepam_gaussian \
    --lattice_dim 1 \
    --max_iterations 10000 \
    --seed 1234 \
    --model cnn2 \
    --baseline cepam \
    --privacy_type gaussian \
    --sigma 0.01


For CEPAM Laplace mechanism:

```md
python main.py \
    --exp_name fl_cepam_laplace \
    --lattice_dim 1 \
    --max_iterations 10000 \
    --seed 1234 \
    --model cnn2 \
    --baseline cepam \
    --privacy_type laplace \
    --b 0.1

For other baselines (fl_privacy_sdq, fl_privacy, fl_sdq, fl):

```md
python main.py \
    --exp_name fl_gaussian_sdq \
    --lattice_dim 1 \
    --seed 1234 \
    --model cnn2 \
    --baseline fl_privacy_sdq \
    --privacy_type gaussian \
    --sigma 0.01

### Testing

For CEPAM Gaussian:

```md
python main.py --exp_name=fl_cepam_gaussian --eval --model cnn2

For CEPAM Laplace:
python main.py --exp_name=fl_cepam_laplace --eval --model cnn2

For other baselines (fl_privacy_sdq, fl_privacy, fl_sdq, fl):
python main.py --exp_name=fl_gaussian_sdq --eval --model cnn2
