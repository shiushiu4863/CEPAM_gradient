# Communication-Efficient and Privacy-Adaptable Mechanism for Federated Learning with gradient updates

## Intrdocution

In this work, we further study CEPAM, a joint mechanism that achieve privacy while compressing gradient updates in FL. CEPAM leverages RSUQ, building upon its capibility to convert quantization distortion into an additive noise term with adjustable variance, independent of the quantized data. Within CEPAM, clients and server can tailor the privacy mechanism. In addition, the inherent natrue of RSUQ as a vector quantizer aids in reducing the compression ratio compared to its scalar counterparts. This repository contains a baisc PyTorch implementation of CEPAM for both Gaussian (CEPAM-Gaussian) and Laplace (CEPAM-Laplace) mechanisms. Additionally, we provide basci implementation of all the baseline mechanisms mentioned in the paper. 

## Usage

This code has been tested on Python 3.13.5, PyTorch 2.7.1 and CUDA 11.5.

## Prerequisite

1. PyTorch=2.7.1: https://pytorch.org
2. scipy
3. tqdm
4. matplotlib
5. torchinfo
6. TensorboardX: https://github.com/lanpa/tensorboardX
