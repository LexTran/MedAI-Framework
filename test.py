from data.dataset import get_loader 

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import SimpleITK as sitk
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai import transforms as tfs
from monai.metrics import DiceMetric
from monai.data import (decollate_batch,
                    load_decathlon_datalist)
from thop import profile

import argparse
import os
import subprocess
import time

# parameters
parser = argparse.ArgumentParser(description='Medical 3D Reconstruction')
parser.add_argument('--resume_path', default=None, help='resume path')
parser.add_argument('--bs', default=1, help='batch size')
parser.add_argument('--output_path', default='./output/', help="save epoch")
parser.add_argument('--dp', default=False, help="whether to use ddp or not")
parser.add_argument('--classes', default=1, help="how many classes to segment")
parser.add_argument('--data_path', default="/home/ubuntu/disk1/TLX/datasets/seg_demo/multi-organ/images/", 
                    help="where you put your data")
parser.add_argument('--mask_path', default="/home/ubuntu/disk1/TLX/datasets/seg_demo/multi-organ/labels/", 
                    help="where you put your mask")
args = parser.parse_args()

# set GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cudnn.enable = True
cudnn.benchmark = True
if args.dp:
    device_ids = [i for i in range(torch.cuda.device_count())]
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
    device_ids = [0]

# loading datasets
batch_size = int(args.bs)
ct_path1 = args.data_path
x_path1 = args.mask_path
ct_path = [ct_path1]
drr_path = [x_path1]
test_loader = get_loader(batch_size, drr_path, ct_path, mode='test')
dim = len(test_loader.dataset[0][0]["volume"].shape)-1
if dim == 2:
    post_fix = ".png"
elif dim == 3:
    post_fix = ".nii.gz"

# set your model
model = UNet(
    dimensions=dim,
    in_channels=1,
    out_channels=int(args.num_classes),
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# whether to use data parallel to support multi GPUs
if args.dp:
    print("Using multi GPUs...")
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
# device = torch.device("cpu")

# calculate flops and params
shape = test_loader.dataset[0][0]["volume"].shape
num_sample = len(test_loader.dataset[0])
flops, params = profile(model, inputs=(torch.randn(num_sample,shape[0],shape[1],shape[2],shape[3]).to(device),))
print('flops: {:.2f}G, params: {:.2f}M'.format(flops/1e9, params/1e6))

# loading checkpoints
checkpoint = torch.load(args.resume_path, map_location=torch.device("cpu"))
if args.resume_path is not None:
    print("Continue training...")
    checkpoint = torch.load(args.resume_path, map_location=torch.device("cpu"))
    if args.dp:
        model.module.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint['net'])
    best_dice_metric = checkpoint['best_dice']
    best_dice_epoch = checkpoint['best_dice_epoch']
else:
    raise ValueError("No checkpoint found at '{}'".format(args.resume_path))

# create needed folders
if os.path.exists(args.output_path) == False:
    os.makedirs(args.output_path)
output_path = args.output_path
if os.path.exists(args.output_path+'/test/') == False:
    os.makedirs(args.output_path+'/test/')
test_output_path = args.output_path+'/test/'

# post-processing
if int(args.classes) == 1:
    post_label = tfs.Compose([tfs.AsDiscrete(threshold=0.5)])
    post_pred = tfs.Compose([tfs.Activations(sigmoid=True),tfs.AsDiscrete(threshold=0.5),
                             tfs.KeepLargestConnectedComponent(),tfs.FillHoles()])
elif int(args.classes) > 1:
    post_label = tfs.Compose([tfs.AsDiscrete(to_onehot=int(args.classes))])
    post_pred = tfs.Compose([tfs.Activations(softmax=True),tfs.AsDiscrete(argmax=True, to_onehot=int(args.classes)),
                             tfs.KeepLargestConnectedComponent(is_onehot=True),
                             tfs.RemoveSmallObjects(min_size=32,independent_channels=True),tfs.FillHoles()])
start = time.time()

# inference
dice_metric = 0
model.eval()
for step, test_sample in enumerate(test_loader):
    with torch.no_grad():
        test_label, test_ct = torch.round(test_sample["label"].float().cuda()), test_sample["volume"].float().cuda()
        test_name = test_sample['name']
        
        # segmentation
        test_seg = sliding_window_inference(test_ct,(32,32,32),4,model,overlap=0.8)
        
        test_labels_list = decollate_batch(test_label)
        test_labels_convert = [post_label(i) for i in test_labels_list]
        test_outputs_list = decollate_batch(test_seg)
        test_output_convert = [post_pred(i) for i in test_outputs_list]
        Dice = DiceMetric(include_background=True, reduction="mean", 
                            get_not_nans=False)(y_pred=test_output_convert, y=test_labels_convert).mean()
        
        # save test results for visualization
        if int(args.classes) == 1:
            save_seg = test_seg.detach().cpu()
            save_seg = save_seg.squeeze(1)
        elif int(args.classes) > 1:
            save_seg = torch.argmax(test_seg, dim=1).detach().cpu()
        if not os.path.exists(test_output_path):
            os.makedirs(test_output_path)
        if not os.path.exists(test_output_path+"/trans_label/"):
            os.makedirs(test_output_path+"/trans_label/")
            for idx in range(test_label.shape[0]):
                label = test_label[idx].detach().cpu().squeeze(0).numpy().astype(np.uint8)
                save_label = sitk.GetImageFromArray(label)
                sitk.WriteImage(save_label, test_output_path+"/trans_label/"+test_name[idx]+'_ori'+post_fix)
        for idx in range(save_seg.shape[0]):
            res_vol = save_seg[idx].numpy().astype(np.uint8)
            save_volume = sitk.GetImageFromArray(res_vol)
            sitk.WriteImage(save_volume, test_output_path+"/"+test_name[idx]+post_fix)
    
    mean_dice = Dice.item()

    print("test dice:{:.4f}".format(mean_dice))

testing_time = time.time() - start
print('Finished Inference')
print('Inference time: {:.4f} seconds'.format(testing_time))
