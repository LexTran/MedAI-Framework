from glob import glob
import os
import math
import numpy as np

import torch
import monai
from monai.data import Dataset,DataLoader,list_data_collate,decollate_batch,pad_list_data_collate
from monai import transforms as tfs
from PIL import Image
import SimpleITK as sitk

class ReconDataset3D(Dataset):
    def __init__(self, drr_dir: list, ct_dir: list, norm=False):
        self.front_drr_list = []
        self.side_drr_list = []
        self.ct_list = []
        self.norm = norm
        self.name_list = []
        # get all subsets
        for sub_drr in drr_dir:
            # obtain all files under dir
            front_dir = os.path.join(sub_drr, 'front')
            front_drr_list = sorted(os.listdir(front_dir))
            # complete path
            front_drr_list = list(map(lambda x: os.path.join(front_dir, x), front_drr_list))
            self.front_drr_list += front_drr_list
            # AP
            side_dir = os.path.join(sub_drr, 'side')
            side_drr_list = sorted(os.listdir(side_dir))
            side_drr_list = list(map(lambda x: os.path.join(side_dir, x), side_drr_list))
            self.side_drr_list += side_drr_list
            
        for sub_ct in ct_dir:
            ct_list = sorted(os.listdir(sub_ct))
            ct_list = list(map(lambda x: os.path.join(sub_ct, x) , ct_list))
            self.ct_list += ct_list
            if len(self.name_list)==0:
                self.name_list = list(map(lambda x: x.split('/')[-1], ct_list))

    def __len__(self):
        return len(self.ct_list)

    # load data as [xray_replication, ct]
    def __getitem__(self, idx):
        front_drr_path = self.front_drr_list[idx]
        side_drr_path = self.side_drr_list[idx]
        ct_path = self.ct_list[idx]

        front = Image.open(front_drr_path)
        if front.mode != 'L':
            front = front.convert('L')
        else:
            front = front
        side = Image.open(side_drr_path)
        if side.mode != 'L':
            side = side.convert('L')
        else:
            side = side

        ct_model = sitk.ReadImage(ct_path, sitk.sitkInt16)
        origin = ct_model.GetOrigin()
        spacing = ct_model.GetSpacing()
        direction = ct_model.GetDirection()

        front_array = np.array(tfs.Resize((128, 128), antialias=True)(front)).astype('float')
        front_array = (front_array-front_array.mean())/front_array.std() # distribution normalization
        front_array = np.expand_dims(front_array, axis=0).repeat(128, axis=0) # (128,128) -> (128,128,128)
        side_array = np.array(tfs.Resize((128, 128), antialias=True)(side)).astype('float')
        side_array = (side_array-side_array.mean())/side_array.std() # distribution normalization
        side_array = np.expand_dims(side_array, axis=2).repeat(128, axis=2)

        ct_array = sitk.GetArrayFromImage(ct_model)
        ori_size = ct_array.shape
        # make sure the ct value is between [-1,1]
        ct_array = ct_array.astype('float')
        # reshape it into 128x128x128
        ct_labels = torch.FloatTensor(ct_array).unsqueeze(0) # (128,128,128) -> (1,128,128,128)
        # ct_labels = Orientation(labels=('LAS'))(ct_labels)
        ct_labels = tfs.Spacing(pixdim=[1, 1, 1])(ct_labels, mode='bilinear')
        ct_labels = tfs.Resize(spatial_size=[128, 128, 128], mode='trilinear')(ct_labels)
        ct_mask = torch.where(ct_labels>-500, True, False)
        
        if self.norm:
            # max_value, min_value = torch.max(ct_array), torch.min(ct_array)
            mean = ct_labels.mean()
            std = ct_labels.std()
            ct_labels = tfs.NormalizeIntensity()(ct_labels)

        # transform the array into tensor
        front_array = np.expand_dims(front_array, axis=0) # (128,128,128) -> (1,128,128,128)
        side_array = np.expand_dims(side_array, axis=0)
        enhanced = np.concatenate((front_array,side_array), axis=0, dtype='float') # (1,128,128,128) -> (2,128,128,128)
        sample = {'drr': enhanced, 'ct': ct_labels, 'mask': ct_mask.int(), 
                'name': self.name_list[idx], 'mean': mean, 'std': std, 'origin': np.array(origin), 
                'spacing': np.array(spacing), 'direction': np.array(direction), 'ori_size': np.array(ori_size)}
        return sample
    
    