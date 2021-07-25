# Self-Attention-Based-Medical-Speech-Recognition
* [Overview](#Overview)
* [Intallation](#Intallation)
    * [Hardware Information](#Hardware-Information)
    * [Software Requirements](#Software-Requirements)
    * [Install Dependencies](#Install-Dependencies)
* [Dataset](#Dataset)
* [Preprocessing](#Preprocessing)
* [Training](#Training)
* [Testing](#Testing)

## Overview
#### The main features:
* In this repository, we have developed a self-attention based medical speech recognition model for recording speeches during nursing shift handovers.
* A nursing handover dataset has been collected. And this dataset contains labeled audio speeches recorded from nursing stations.
#### Base:
* The model is based on [Espnet Toolkit](https://github.com/espnet/espnet), and the **changes are in [CHANGELOG.md](CHANGELOG.md)**, and you can check it for all the details.

## Intallation
### Hardware Information
These are the hardware inforamation we used to train or test our model.
* CPU: Intel Core i7-9700K
* GPU: NVIDIA GTX 2080 super 8GB
* RAM: 32 GB
### Software Requirements
These are software and framework versions.
* OS: **Ubuntu 16.04 (Note: Not support for windows)**
* CUDA: 10.1
* cuDNN: 7.6.5
* Python: 3.7
* Pytorch: 1.4.0
### Install Dependencies
#### 1. **Make sure to install cmake3, sox, sndfile, ffmpeg, flac on Ubuntu OS**:
```
sudo apt-get install cmake sox libsndfile1-dev ffmpeg flac
```
#### 2. Install **Kaldi for speech preprocessing**:
```
cd <any-place>
git clone https://github.com/kaldi-asr/kaldi
cd kaldi/tools
make -j 4
./extras/install_openblas.sh
sudo ./extras/install_mkl.sh
sudo apt-get install libatlas-base-dev
cd ..
cd kaldi/src
./configure --use-cuda=no
make -j clean depend; make -j 4
```
#### 3. Install **ESPnet Toolkit**:
##### (1) Git clone espnet:
```
cd <any-place>
git clone https://github.com/espnet/espnet
```
##### (2) Put Compiled Kalfi under espnet/tools:
```
cd espnet/tools
ln -s kaldi .
```
##### (3) Set python environment using Anaconda
```
cd espnet/tools
CONDA_TOOLS_DIR=/home/ee303/anaconda
./setup_anaconda.sh ${CONDA_TOOLS_DIR} espnet 3.8
```
#### 4. Install Espnet
```
cd espnet/tools
make
make TH_VERSION=1.4.0 CUDA_VERSION=10.1
```
## Dataset
This model supports for **Chinese Medical Speech Corpus (sChiMeS)** and **Punctuation Chinese Medical Speech Corpus (psChiMeS)**dataset for training and testing. If you are using other dataset, you have to reconstruct the dataset directories refer to the following descriptions.

**sChiMeS and dataset is released on https://iclab.ee.ntust.edu.tw/datasets/**, so you can download it and use this baseline by the following descriptions.

## Preprocessing

## Training

## Testing

