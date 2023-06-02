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
if args.dp:
    device_ids = [i for i in range(torch.cuda.device_count())]
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
    cudnn.benchmark = False
    device_ids = [1]

# loading datasets
batch_size = int(args.bs)
ct_path1 = "/home/ubuntu/disk1/TLX/datasets/Task501_Spine/imagesTr/"
x_path1 = "/home/ubuntu/disk1/TLX/datasets/Task501_Spine/Xray/"
ct_path = [ct_path1]
drr_path = [x_path1]

test_loader = get_loader(batch_size, drr_path, ct_path, mode='test')
shape = test_loader.dataset[0][0]["volume"].shape
num_sample = len(test_loader.dataset[0])

model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=int(args.num_classes),
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

if args.dp:
    print("Using multi GPUs...")
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
# device = torch.device("cpu")
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

start = time.time()

# validation
# reset metrics for each validation
dice_metric = 0
psnr_metric = 0
ssim_metric = 0
mae_metric = 0
mse_metric = 0
model.eval()
for step, test_sample in enumerate(test_loader):
    with torch.no_grad():
        test_label, test_ct = test_sample["label"].float().cuda(), test_sample["volume"].float().cuda()
        test_name = test_sample['name']
        
        # segmentation
        test_seg = sliding_window_inference(test_ct,(96,96,96),4,model,overlap=0.8)
        
        test_labels_list = decollate_batch(test_label)
        test_labels_convert = [tfs.AsDiscrete(to_onehot=14)(val_label_tensor) for val_label_tensor in val_labels_list]
        test_outputs_list = decollate_batch(test_seg)
        test_output_convert = [tfs.AsDiscrete(argmax=True, to_onehot=14)(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        Dice = DiceMetric(include_background=True, reduction="mean", 
                            get_not_nans=False)(y_pred=test_output_convert, y=test_labels_convert).mean()
        
        # save validation results for visualization
        save_seg = torch.argmax(test_seg, dim=1).detach().cpu()
        if not os.path.exists(test_output_path):
            os.makedirs(test_output_path)
        for idx in range(save_seg.shape[0]):
            res_vol = save_seg[idx].numpy()
            save_volume = sitk.GetImageFromArray(res_vol)
            sitk.WriteImage(save_volume, test_output_path+"/"+test_name[idx]+".nii.gz")
    
    mean_dice = Dice.item()

    print("test dice:{:.4f}".format(mean_dice))

testing_time = time.time() - start
print('Finished Inference')
print('Inference time: {:.4f} seconds'.format(testing_time))
