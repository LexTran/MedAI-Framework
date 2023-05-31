from data.dataset import get_loader 
from utils.metric import compute_metrics

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import SimpleITK as sitk
from monai.networks.nets import UNet

import argparse
import os
import subprocess
import time

# parameters
parser = argparse.ArgumentParser(description='Medical 3D Reconstruction')
parser.add_argument('--resume_path', default=None, help='resume path')
parser.add_argument('--bs', default=4, help='batch size')
parser.add_argument('--output_path', default='./output/', help="save epoch")
args = parser.parse_args()

# set GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

bx2s_net = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# if torch.cuda.device_count() > 1:
#     print("Using multi GPUs...")
#     vae = vae.to(device)
#     vae = torch.nn.DataParallel(vae, device_ids=device_ids)
# else:
#     vae.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

bx2s_net.to(device)

# loading checkpoints
checkpoint = torch.load(args.resume_path, map_location=torch.device("cpu"))
bx2s_net.load_state_dict(checkpoint['net'])

# create needed folders
if os.path.exists(args.output_path) == False:
    os.makedirs(args.output_path)
output_path = args.output_path

start = time.time()

loss_fn = nn.MSELoss(reduction='mean')

# validation
# reset metrics for each validation
dice_metric = 0
psnr_metric = 0
ssim_metric = 0
mae_metric = 0
mse_metric = 0
bx2s_net.eval()
for step, test_sample in enumerate(test_loader):
    with torch.no_grad():
        test_xct, test_ct = test_sample["dim_enhance"].float().cuda(), test_sample["ct"].float().cuda()
        test_name = test_sample['name']
        
        # reconstruction
        test_recon = bx2s_net(test_xct).detach()
        
        # compute metrics for current iteration
        mae, mse, psnr, ssim, dice = compute_metrics(test_recon, test_ct) 
        mae_metric += mae
        mse_metric += mse
        psnr_metric += psnr
        ssim_metric += ssim
        dice_metric += dice

        # save validation results for visualization
        res_vol = test_recon.float().cpu().numpy()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for idx in range(res_vol.shape[0]):
            pred = res_vol[idx].squeeze(0)
            save_volume = sitk.GetImageFromArray(pred)
            sitk.WriteImage(save_volume, output_path+"/"+test_name[idx]+".nii.gz")
    
    freq_metric = freq_metric/len(test_loader)
    mae_metric = mae_metric/len(test_loader)
    mse_metric = mse_metric/len(test_loader)
    psnr_metric = psnr_metric/len(test_loader)
    ssim_metric = ssim_metric/len(test_loader)
    dice_metric = dice_metric/len(test_loader)

    print("val mse:{:.4f}; val dice:{:.4f}; val psnr:{:.4f}; val ssim:{:.4f}; val mae:{:.4f};".format(
        mse_metric,dice_metric,psnr_metric,ssim_metric,mae_metric))

testing_time = time.time() - start
print('Finished Inference')
print('Inference time: {:.4f} seconds'.format(testing_time))
