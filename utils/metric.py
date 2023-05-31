import torch
import sklearn.metrics
import numpy as np
from chamfer_distance import ChamferDistance
import mcubes
from monai.transforms.post.array import AsDiscrete
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import mean_absolute_error as mae
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import mean_squared_error as mse
from torchmetrics.functional import dice

def ap(predict, gt):
    predict_clone = predict.clone().cpu().numpy()
    gt_clone = gt.clone().cpu().numpy()

    batch_size = predict_clone.shape[0]
    precisions = []
    for i in range(batch_size):
        predict_one = predict_clone[i, ...].flatten()
        gt_one = gt_clone[i, ...].flatten()
        precisions.append(sklearn.metrics.average_precision_score(gt_one, predict_one))
    avg_precision = np.array(precisions).mean()
    return avg_precision.item()

def cd(predict, gt):
    predict_clone = predict.clone().cpu().numpy()
    gt_clone = gt.clone().cpu().numpy()

    batch_size = predict_clone.shape[0]
    cd_metrics = 0.
    for i in range(batch_size):
        gt_vertices, _ = mcubes.marching_cubes(gt_clone[i].squeeze(), 0)
        predict_vertices, _ = mcubes.marching_cubes(predict_clone[i].squeeze(), 0)
        gt_pc = torch.tensor(gt_vertices).type(torch.float32).unsqueeze(dim=0).cuda()
        predict_pc = torch.tensor(predict_vertices).type(torch.float32).unsqueeze(dim=0).cuda()

        cal_cd = ChamferDistance()
        dist1, dist2, idx1, idx2 = cal_cd(gt_pc, predict_pc)
        cd_metrics += torch.mean(dist1) + torch.mean(dist2)
    return (cd_metrics / batch_size).item()

def compute_metrics(val_recon_g, val_ct):
    mae_metric = mae(val_recon_g, val_ct)
    mse_metric = mse(val_recon_g, val_ct)
    psnr_metric = psnr(val_recon_g, val_ct)
    val_recon_g = AsDiscrete(argmax=True)(val_recon_g)
    val_ct = AsDiscrete(argmax=True)(val_ct)
    ssim_metric = ssim(val_recon_g, val_ct)
    dice_metric = dice(val_recon_g.int(), val_ct.int())
    return mae_metric, mse_metric, psnr_metric, ssim_metric, dice_metric