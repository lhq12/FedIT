# ICML 2026: Powerful and Theoretically Guaranteed Independence Testing on Heterogeneous Federated Clients

## Introduction

This is the official PyTorch implementation of ICML 2026 paper "[Powerful and Theoretically Guaranteed Independence Testing on Heterogeneous Federated Clients](https://openreview.net/forum?id=hH0EknXrgc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2026%2FConference%2FAuthors%23your-submissions))".

## Preparation

1. Create conda environment:
   
```shell
conda create -n fedit python=3.8
conda activate fedit
```

2. Install dependencies:
```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install scipy
conda install main::matplotlib
```

## Run the code

Experiments on synthetic data involved in the paper have been included in demo.ipynb, allowing the results to be reproduced directly.

## Citation

If this code is useful for your research, please consider citing:

  ```shell
@inproceedings{
anonymous2026powerful,
title={Powerful and Theoretically Guaranteed Independence Testing on Heterogeneous Federated Clients},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=hH0EknXrgc}
}

  ```
