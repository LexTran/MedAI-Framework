# MedSeg-Framework
This repository provides an universal pipeline for medical image segmentation with the support of Pytorch &amp; MONAI.

## Requirement

Here we give the environment tested on our device, but we havn't tried on different versions.
- python == 3.10.4
- pytorch == 1.12.1
- cudatoolkit == 11.6.0
- tensorboard == 2.8.0
- torchmetrics == 0.11.2
- pillow == 9.2.0
- opencv-python == 4.7.0.68
- monai == 1.1.0
- simpleitk == 2.2.0

## Training

To train your own network, you need to create a model file under `\network` and use it in `train.py`

Also, you will need to modify the data path in `train.py` 
>52  ct_path1 = "/home/ubuntu/disk1/TLX/datasets/seg_demo/images/"

>53  mask_path1 = "/home/ubuntu/disk1/TLX/datasets/seg_demo/labels/"

The command to run the file is as follow:
```
python train.py --epoch=100 --lr=0.01 --board=<where to put your tensorboard log> --save_path=<where to save your model> --output_path=<where to save your results for visualization> --dp=True --classes=1
```
in this case, `dp` decides whether to use data parallel to support multi-gpus and `classes` decides how many classes to segment.

## To Do
- [ ] Add Multi-task segmentation
- [ ] 
