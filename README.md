# Mutual Distillation Driven Dual-Space Matching for Visible–Infrared Person Re-Identification

Welcome to use the code from our paper **"Domain Shifting: A Generalized Solution for Heterogeneous Cross-Modality Person Re-Identification"**.

## Usage

### Environment
- Python 3.10+
- PyTorch >= 2.0.1

### Dataset
Prepare the datasets 




### Preprocessing
Preprocess SYSU-MM01 and LLCM datasets for faster training:

```bash
python pre_process_sysu.py
python pre_process_llcm.py
```bash

### Training
SYSU-MM01:


python train.py --dataset sysu --lr 0.2
RegDB:


bash train_regdb.sh
LLCM:


python train.py --dataset llcm --lr 0.2
Tip: Setting --max_epoch=80 can reduce training time by ~20% with negligible performance loss:


python train.py --dataset sysu --lr 0.2 --max_epoch=80
Testing
You can find the training logs and checkpoints in Google Drive. Testing commands:

SYSU-MM01 (all search):


python test.py --dataset sysu --resume sysu_p6_n4_lr_0.2_seed_0_best.pth --mode all
SYSU-MM01 (indoor search):


python test.py --dataset sysu --resume sysu_p6_n4_lr_0.2_seed_0_best.pth --mode indoor
RegDB (Infrared to Visible):


python test.py --dataset regdb --tvsearch 0
RegDB (Visible to Infrared):


python test.py --dataset regdb --tvsearch 1
LLCM (Infrared to Visible):


python test.py --dataset llcm --resume llcm_p6_n4_lr_0.2_seed_0_best.pth --tvsearch 0
LLCM (Visible to Infrared):


python test.py --dataset llcm --resume llcm_p6_n4_lr_0.2_seed_0_best.pth --tvsearch 1


Contact
For questions, contact: dongcanliu@stu.cqut.edu.cn

