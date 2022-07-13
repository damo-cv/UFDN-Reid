![Python >=3.5](https://img.shields.io/badge/Python->=3.5-yellow.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)

# [ECCV2022] Unstructured Feature Decoupling for Vehicle Re-Identification (UFDN)

## News
- 2022.7  We release the code of UFDN.

## Pipeline

![framework](figs/overview.pdf)



## Requirements

### Installation

```bash
pip install -r requirements.txt
(we use /torch 1.7.1 /torchvision 0.8.2 /timm 0.3.2 /cuda 11.0 / 16G or 32G V100 for training and evaluation.)
```

### Prepare Datasets

```bash
mkdir data
```

Download the vehicle datasets [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html), [VeRi-776](https://github.com/JDAI-CV/VeRidataset), 
Then unzip them and rename them under the directory like

```
data
└── VeRi
    └── images ..
└── VehicleID
    └── images ..
└── VERI-WILD
    └── images ..

```

### Prepare Res50 or Swin Pre-trained Models

You need to download the ImageNet pretrained transformer model : [Res50](https://download.pytorch.org/models/resnet50-19c8e357.pth), [Swin-tiny](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth).

## Training

We utilize 1  GPU for training.

```
sh experiments/train_res50_submit.sh or train_swin_UFDN.sh
```

## Acknowledgement

Codebase from [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) , [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)


## Contact

If you have any question, please feel free to contact us. E-mail: [qianwen2018@ia.ac.cn](qianwen2018@ia.ac.cn) , [haoluocsc@zju.edu.cn](mailto:haoluocsc@zju.edu.cn)