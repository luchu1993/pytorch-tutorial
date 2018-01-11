# pytorch-tutorial

This tutorail will give some commom neural network models, such as:
1. Feed Forward Network(FNN)
2. Convolution Neural Network (CNN)
3. Recurrent Neural Network (RNN)
4. Generative Adversarial Networks (GAN)

We will use pytorch to implement it.

## Build develop environment

### Install anaconda
Download the anaconda on https://www.anaconda.com/download/, and install it.

### Install CUDA8.0
Download the CUDA toolkit https://pan.baidu.com/s/1c2ZwOA8, then install it.

### Install pytorch package 

#### Windows
Pytorch 0.3 is not support Windows10, download the third party package: https://pan.baidu.com/s/1c3Sl5du

and then using `conda` to install it.
```bash
conda install numpy mkl cffi
conda install --offline pytorch_legacy-0.3.0-py36_0.3.0cu80.tarbz2
```
to test the installation, input it in python:
```python
import torch
```
if there is no error, it's ok.

#### Linux or MacOS
For Linux or MaxOS, we can find the install command on the pytorch website: http://pytorch.org/
1. Linux:
```bash
conda install pytorch-cpu torchvision -c pytorch
```
2. MacOS
```bash
conda install pytorch torchvision -c pytorch 
# macOS Binaries dont support CUDA, install from source if CUDA is needed
```
