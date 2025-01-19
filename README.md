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
conda activate openmmlab
```

### 2. Install PyTorch and Dependencies
Next, install PyTorch version 1.9.0 along with torchvision and the necessary CUDA toolkit:

```bash
conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 -c pytorch
```

### 3. Install OpenMMLab and Other Dependencies
Now install the OpenMMLab tools and libraries using mim (OpenMMLab installation manager):

```bash
pip install openmim==0.3.9
mim install mmengine==0.10.3
mim install mmcv==2.0.1
mim install mmdet==3.2.0
```

### 4. Install Additional Libraries
Install other required libraries:

```bash
pip install timm
pip install segmentation-models-pytorch
mim install mmyolo==0.6.0
```

### 5. Verify Installation
To verify your installation, you can run the following Python commands:

```python
import torch
import mmcv
import mmdet
import mmyolo

print(f"PyTorch version: {torch.__version__}")
print(f"MMCV version: {mmcv.__version__}")
print(f"MMDetection version: {mmdet.__version__}")
print(f"MMYOLO version: {mmyolo.__version__}")
```

### 6. Download Checkpoint

You need to download the checkpoints from the official MMYOLO repository. You can download the YOLOv8 checkpoint from the following link: https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8

### 7. Notes

- The versions specified are known to work together
- If you need to use different versions, make sure to check the compatibility matrix in the OpenMMLab documentation
- For GPU support, ensure you have the correct NVIDIA drivers installed
