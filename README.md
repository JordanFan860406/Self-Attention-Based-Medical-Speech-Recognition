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
* GPU: NVIDIA GTX 2080 super
* RAM: 8GB
### Software Requirements
These are software and framework versions.
* OS: **Ubuntu 16.04 (Note: Not support for windows)**
* CUDA: 10.1
* cuDNN: 7.6.5
* Python: 3.7
* Pytorch: 1.4.0
### Install Dependencies
