# MedAI-Framework
This repository provides an universal pipeline for medical image segmentation with the support of Pytorch &amp; MONAI.

* <a href="#require">Requirement</a>
* <a href="#train">Training</a>
* <a href="#test">Testing</a>
* <a href="#multi">Multi-label Segmentation</a>
* <a href="#recon">Reconstruction</a>
* <a href="#regis">Registration</a>
* <a href="#todo">TO DO</a>

## <div id="require">Requirement</div>

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
- thop == 0.1.1.post2209072238

## <div id="train">Training</div>

To train your own network, you need to create a model file under `\network` and use it in `segmentation.py`

The command to run training is as follow:
```
python segmentation.py --mode=train --epoch=<how many epoch you want to train> --lr=<learning rate> --board=<where to put your tensorboard log> --save_path=<where to save your model> --output_path=<where to save your results for visualization> --dp=<whether to use data parallel> --classes=<number of your segmentation targets> --data_path=<your data path> --mask_path=<your label path>
```

## <div id="test">Testing</div>

To test the well-trained network, you can run testing as follow:
```
python segmentation.py --mode=test --resume_path=<where to load your model> --output_path=<where to save your results for visualization> --dp=<whether to use data parallel> --classes=<number of your segmentation targets> --data_path=<your data path> --mask_path=<your label path>
```

## <div id="multi">Multi-label segmentaion</div>

Our framework supports multi-organ segmentaion, which in many cases may encounter with multiple label files. We have written a script to convert multiple labels into one label, you can find it under `\utils`, named `label_convert.py`.

## <div id="recon">Reconstruction</div>

We also support 3D reconstruction, to do so, you need to prepare your into 3 directories, including `datasets\ct\`, `datasets\drr\front\` and `datasets\drr\side\`.

To train your reconstruction net, use following command:
```
python reconstruction.py --mode=train --epoch=<how many epoch you want to train> --lr=<learning rate> --board=<where to put your tensorboard log> --save_path=<where to save your model> --output_path=<where to save your results for visualization> --dp=<whether to use data parallel> --data_path=<your data path> --drr_path=<your drr path>
```

## <div id="regis">Registration</div>

The medical image registration will be supported soon.

Ideally, you can train your registration net using following command:
```
python registration.py --mode=train --epoch=<how many epoch you want to train> --lr=<learning rate> --board=<where to put your tensorboard log> --save_path=<where to save your model> --output_path=<where to save your results for visualization> --dp=<whether to use data parallel> --fixed_path=<your fixed data path> --moving_path=<your moving data path>
```

## Tips

We have provided many useful tools to help you perform data format convertion such as, `mhd -> nii`, `ima -> nii`, `nii -> stl`. Those tools are put under `utils`, have fun with them your way.

## <div id="todo">To Do</div>
- [x] Add Multi-task segmentation
- [x] Convert multi labels into one label
- [x] Extend this framework to enable reconstruction
- [ ] Semi-supervised support
- [ ] Accelerate 3d convolution using self-defined operator in Taichi
- [ ] Extend this framework to enable registration and reconstruction

