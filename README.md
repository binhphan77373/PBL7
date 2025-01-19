# PBL7
Để cài đặt các gói phần mềm cần thiết, bạn có thể sử dụng file requirements.txt hoặc environment.yaml.

Sử dụng requirements.txt
pip install -r requirements.txt

Sử dụng environment.yaml
conda env create -f environment.yaml

Tải checkpoint
Bạn cần tải các checkpoint từ repository chính thức của MMYOLO. Bạn có thể tải checkpoint YOLOv8 từ đường dẫn sau:

https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8


# OpenMMLab Environment Setup

This guide walks you through the steps to set up the OpenMMLab environment with the necessary dependencies.

## Requirements

- Conda
- Python 3.8

## Steps to Set Up

### 1. Create and Activate the Conda Environment
First, create a new conda environment with Python 3.8:

```bash
conda create --name openmmlab python=3.8 -y

### 2. Install PyTorch and Dependencies
Next, install PyTorch version 1.9.0 along with torchvision and the necessary CUDA toolkit:
```bash
conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 -c pytorch
