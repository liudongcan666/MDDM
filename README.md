# Mutual Distillation Driven Dual-Space Matching for Visible–Infrared Person Re-Identification

Welcome to use the code from our paper **"Mutual Distillation Driven Dual-Space Matching for Visible–Infrared Person Re-Identification"**.



##  Environment

- Python 3.10+
- PyTorch >= 2.0.1



##  Dataset

SYSU-MM01, RegDB, and LLCM

##  Preprocessing

Preprocess SYSU-MM01 and LLCM datasets for faster training:

```bash
python pre_process_sysu.py
python pre_process_llcm.py
```bash

Training and Testing

We provide convenient shell scripts for training and testing. You can directly follow the scripts or run the commands manually.

# Using the training script
bash train_sysu.sh

##  Requirements
we use single RTX3090 24G GPU for training and evaluation.
