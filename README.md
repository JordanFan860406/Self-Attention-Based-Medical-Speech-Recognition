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
##### (1) Git clone our github:
```
cd <any-place>
git clone https://github.com/JordanFan860406/Self-Attention-Based-Medical-Speech-Recognition.git
```
##### (2) Put Compiled Kalfi under espnet/tools:
```
cd Self-Attention-Based-Medical-Speech-Recognition/espnet/tools
ln -s kaldi .
```
##### (3) Set python environment using Anaconda
```
cd Self-Attention-Based-Medical-Speech-Recognition/espnet/tools
CONDA_TOOLS_DIR=/home/ee303/anaconda
./setup_anaconda.sh ${CONDA_TOOLS_DIR} espnet 3.8
```
#### 4. Install Espnet
```
cd Self-Attention-Based-Medical-Speech-Recognition/espnet/tools
make
make TH_VERSION=1.4.0 CUDA_VERSION=10.1
```
## Corpus
This model supports for **Chinese Medical Speech Corpus (sChiMeS)** and **Punctuation Chinese Medical Speech Corpus (psChiMeS)**dataset for training and testing. If you want to use them, you can download on our [Google Drive](https://drive.google.com/drive/folders/1AVhkHPOLvZMwWBNqI5kXC85PsOmB-032?usp=sharing). Note: This corpus format is only for this Model (the corpus is divided by recordist id).

If you want to download the original formats of sChiMeS and psChiMeS, **sChiMeS and dataset is released on https://iclab.ee.ntust.edu.tw/datasets/**.

## Preprocessing
You only download our [Dataset](#Dataset) and unzip the file, then move the corpus file to **`espnet/egs/sChiMeS-14/data/chimes_14` for sChiMeS-14** or **`espnet/egs/psChiMeS-14/data/chimes_14` for psChiMeS-14**.

## Training

## Testing

