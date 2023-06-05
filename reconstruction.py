from torch.utils.tensorboard import SummaryWriter
from data.reconDataset import ReconDataset3D
from network.PSR import ReconNet
from utils.recon_metric import compute_metrics

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
import SimpleITK as sitk
from monai.optimizers import WarmupCosineSchedule
from monai import transforms as tfs
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from thop import profile

import argparse
import os
import subprocess
import time
import math

# parameters
parser = argparse.ArgumentParser(description='Medical 3D Reconstruction')
parser.add_argument('--resume_path', default=None, help='resume path')
parser.add_argument('--epoch', default=100, help='training epoch')
parser.add_argument('--bs', default=1, help='batch size')
parser.add_argument('--lr', default=0.01, help="learning rate")
parser.add_argument('--l1', default=1, help="lambda1 for reconstruction loss")
parser.add_argument('--board', default='./runs', help="tensorboard path")
parser.add_argument('--save_path', default='./checkpoints/', help="save path")
parser.add_argument('--output_path', default='./output/', help="save epoch")
parser.add_argument('--dp', default=False, help="whether to use ddp or not")
parser.add_argument('--data_path', default="../../datasets/seg_demo/multi-organ/images/", 
                    help="where you put your data")
parser.add_argument('--drr_path', default="../../datasets/seg_demo/multi-organ/labels/", 
                    help="where you put your drr images")
parser.add_argument('--amp', default=False, help="whether to use amp or not")
parser.add_argument('--mode', default="train", help="train or test")
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
drr_path1 = args.drr_path
ct_path = [ct_path1]
drr_path = [drr_path1]
dataset = ReconDataset3D(drr_path, ct_path, norm=True)
if args.mode == "train":
    length = len(dataset) 
    len_list = [math.ceil(0.9*length), math.floor(0.1*length)]
    # random split train/val for k-fold cross validation
    train_set, val_set = torch.utils.data.random_split(dataset,lengths=len_list,generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
elif args.mode == "test":
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
post_fix = ".nii.gz"

# set your model
model = ReconNet(in_channels=2,out_channels=128)

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
shape = train_loader.dataset[0]["drr"].shape
num_sample = len(train_loader.dataset[0])
flops, params = profile(model, inputs=(torch.randn(num_sample,shape[0],shape[1],shape[2],shape[3]).to(device),))
print('flops: {:.2f}G, params: {:.2f}M'.format(flops/1e9, params/1e6))

# create needed folders
if os.path.exists(args.board) == False:
    os.makedirs(args.board)
if os.path.exists(args.output_path) == False:
    os.makedirs(args.output_path)
output_path = args.output_path
if os.path.exists(args.output_path+'/val/') == False:
    os.makedirs(args.output_path+'/val/')
val_output_path = args.output_path+'/val/'
if os.path.exists(args.output_path+'/test/') == False:
    os.makedirs(args.output_path+'/test/')
test_output_path = args.output_path+'/test/'
if os.path.exists(args.save_path) == False:
    os.makedirs(args.save_path)
save_path = args.save_path

loss_fn = MSELoss(reduction='mean')

start = time.time()

def train():
    # loading checkpoints
    if args.resume_path is not None:
        print("Continue training...")
        checkpoint = torch.load(args.resume_path, map_location=torch.device("cpu"))
        if args.dp:
            model.module.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint['net'])
        epoch_start = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr'] = float(args.lr)
    else:
        print("Beginning epoch...")
        epoch_start = 0

    # optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    scheduler = WarmupCosineSchedule(optimizer, 
                                    warmup_steps=10*len(train_loader), 
                                    t_total=int(args.epoch)*len(train_loader))
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    # training settings
    val_interval = 10
    save_interval = 25

    # tensorboard
    writer = SummaryWriter(args.board)

    # init optimizer for scheduler
    optimizer.zero_grad()
    optimizer.step()

    # training loop
    for epoch in range(epoch_start, epoch_start+int(args.epoch)):
        running_loss = 0.0
        
        model.train()
        scheduler.step()
        for step, sample in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(False):
                drr = sample['drr']
                ct = sample['volume']
                drr = drr.float().to(device)
                ct = ct.float().to(device)
                name = sample['name']
                with torch.cuda.amp.autocast():
                    rec = model(drr)
                    rec_loss = int(args.l1)*loss_fn(rec, ct)
                loss = rec_loss

                if args.amp:
                    scaler.scale(loss).backward()
                    # perform gradient clipping in case of gradient explosion 
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                    # in case there is nan in results
                    assert not torch.any(torch.isnan(rec)), "nan in seg"
                    running_loss += loss.item()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                else:
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

        writer.add_scalar('average loss: {:%.4f}', running_loss/len(train_loader), epoch+1)
        print('epoch %d average loss: %.4f' % (epoch+1, running_loss/len(train_loader)))

        # validation
        if (epoch + 1) % val_interval == 0:
            # reset metrics for each validation
            # dice_metric.reset()
            model.eval()
            for step, val_sample in enumerate(val_loader):
                with torch.no_grad():
                    val_drr, val_ct = val_sample["drr"].float().cuda(), val_sample["volume"].float().cuda()
                    val_name = val_sample['name']
                    val_mean = val_sample['mean'].float()
                    val_std = val_sample['std'].float()
                    val_mask = val_sample['mask']
                    
                    # segmentation
                    with torch.cuda.amp.autocast():
                        val_rec = model(val_drr).detach()
                    
                    mae, mse, psnr, ssim, dice = compute_metrics(val_rec.cpu(), val_ct.cpu(), val_mask, val_mean, val_std) 
                    mae_metric += mae
                    mse_metric += mse
                    psnr_metric += psnr
                    ssim_metric += ssim
                    dice_metric += dice

                    # save validation results for visualization
                    res_vol = val_rec.detach().float().cpu()
                    if not os.path.exists(val_output_path+str(epoch+1)):
                        os.makedirs(val_output_path+str(epoch+1))
                    if not os.path.exists(val_output_path+"/trans_label/"):
                        os.makedirs(val_output_path+"/trans_label/")
                        for idx in range(val_ct.shape[0]):
                            label = val_ct[idx].detach().cpu().squeeze(0).numpy().astype(np.uint8)
                            save_label = sitk.GetImageFromArray(label)
                            sitk.WriteImage(save_label, val_output_path+"/trans_label/"+val_name[idx]+'_ori'+post_fix)
                    for idx in range(res_vol.shape[0]):
                        res_vol = res_vol[idx].numpy().astype(np.uint8)
                        save_volume = sitk.GetImageFromArray(res_vol)
                        sitk.WriteImage(save_volume, val_output_path+str(epoch+1)+"/"+val_name[idx]+post_fix)
            
            mae_metric = mae_metric/len(val_loader)
            mse_metric = mse_metric/len(val_loader)
            psnr_metric = psnr_metric/len(val_loader)
            ssim_metric = ssim_metric/len(val_loader)
            dice_metric = dice_metric/len(val_loader)

            print("val mse:{:.4f}; val dice:{:.4f}; val psnr:{:.4f}; val ssim:{:.4f}; val mae:{:.4f}".format(
            mse_metric,dice_metric,psnr_metric,ssim_metric,mae_metric))
            writer.add_scalar('val mse:{:%.4f}',mse_metric,epoch + 1)
            writer.add_scalar('val dice:{:%.4f}',dice_metric,epoch + 1)
            writer.add_scalar('val psnr:{:%.4f}',psnr_metric,epoch + 1)
            writer.add_scalar('val ssim:{:%.4f}',ssim_metric,epoch + 1)
            writer.add_scalar('val mae:{:%.4f}',mae_metric,epoch + 1)
        
        # save checkpoints
        if args.dp:
            state = {
                'net': model.module.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
            }
        else:
            state = {
                'net': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
            }

        if (epoch+1) % save_interval == 0:
            torch.save(state, f'{save_path}/{epoch + 1}.pth') # save model and parameters
            print('Saving epoch %d model ...' % (epoch + 1))

    training_time = time.time() - start
    writer.close()
    print('Finished Training')
    print('Training time: {:.4f} seconds'.format(training_time))

def test():
    model.eval()
    for step, test_sample in enumerate(test_loader):
        with torch.no_grad():
            test_drr, test_ct = test_sample["drr"].float().cuda(), test_sample["volume"].float().cuda()
            test_name = test_sample['name']
            test_mean = test_sample['mean'].float()
            test_std = test_sample['std'].float()
            test_mask = test_sample['mask']
            
            # segmentation
            test_rec = model(test_drr).detach()
            
            mae, mse, psnr, ssim, dice = compute_metrics(test_rec.cpu(), test_ct.cpu(), test_mask, test_mean, test_std) 
            mae_metric += mae
            mse_metric += mse
            psnr_metric += psnr
            ssim_metric += ssim
            dice_metric += dice
            
            # save test results for visualization
            res_vol = test_rec.detach().float().cpu()
            if not os.path.exists(test_output_path):
                os.makedirs(test_output_path)
            if not os.path.exists(test_output_path+"/trans_label/"):
                os.makedirs(test_output_path+"/trans_label/")
                for idx in range(test_ct.shape[0]):
                    label = test_ct[idx].detach().cpu().squeeze(0).numpy().astype(np.uint8)
                    save_label = sitk.GetImageFromArray(label)
                    sitk.WriteImage(save_label, test_output_path+"/trans_label/"+test_name[idx]+'_ori'+post_fix)
            for idx in range(res_vol.shape[0]):
                res_vol = res_vol[idx].numpy().astype(np.uint8)
                save_volume = sitk.GetImageFromArray(res_vol)
                sitk.WriteImage(save_volume, test_output_path+"/"+test_name[idx]+post_fix)
        
        mae_metric = mae_metric/len(test_loader)
        mse_metric = mse_metric/len(test_loader)
        psnr_metric = psnr_metric/len(test_loader)
        ssim_metric = ssim_metric/len(test_loader)
        dice_metric = dice_metric/len(test_loader)

        print("test mse:{:.4f}; test dice:{:.4f}; test psnr:{:.4f}; test ssim:{:.4f}; test mae:{:.4f}".format(
        mse_metric,dice_metric,psnr_metric,ssim_metric,mae_metric))

    testing_time = time.time() - start
    print('Finished Inference')
    print('Inference time: {:.4f} seconds'.format(testing_time))

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()