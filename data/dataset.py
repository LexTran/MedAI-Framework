from glob import glob
import os
import math
import numpy as np

import torch
from monai.data import DataLoader,list_data_collate,decollate_batch
import monai
from monai import transforms as tfs

def get_loader(bs, label_path: list, ct_path: list, mode="train"):
    batch_size = bs
    ct_set = []
    label_set = []
    xray_side_set = []
    for i in range(len(label_path)):
        ct_set += sorted(glob(os.path.join(ct_path[i], "*.nii.gz")))
        label_set += sorted(glob(os.path.join(label_path[i], "*.nii.gz")))
    data_name = [os.path.basename(i).split(".nii.gz")[0] for i in ct_set]

    base_files = [{"volume": volume, "label": label, "name": name} for volume, label, name in zip(ct_set, label_set, data_name)]
    length = len(base_files) 
    len_list = [math.floor(0.9*length), math.ceil(0.1*length)]
    train_files, val_files = torch.utils.data.random_split(base_files,lengths=len_list,generator=torch.Generator().manual_seed(42))

    # how to load the data
    if mode == "train":
        train_transforms = tfs.Compose([
            tfs.LoadImaged(keys=["volume","label"], reader="NibabelReader"),
            tfs.EnsureChannelFirstd(keys=["volume","label"]),
            tfs.Spacingd(keys=["volume","label"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            tfs.Resized(keys=["volume","label"], spatial_size=[128,128,128]),
            tfs.NormalizeIntensityd(keys=["volume"]),
            tfs.RandSpatialCropd(keys=["volume","label"], roi_size=[96,96,96], random_size=False),
            tfs.RandRotated(keys=["volume","label"],range_x=[0.4,0.4],range_y=[0.4,0.4],range_z=[0.4,0.4],prob=0.5,
                            keep_size=True,mode="bilinear",padding_mode="zeros"),
            tfs.RandFlipd(keys=["volume","label"],spatial_axis=0,prob=0.1),
            tfs.RandFlipd(keys=["volume","label"],spatial_axis=1,prob=0.1),
            tfs.RandFlipd(keys=["volume","label"],spatial_axis=2,prob=0.1),
        ])
        train_set = monai.data.Dataset(data=train_files, transform=train_transforms)
        val_transforms = tfs.Compose([
            tfs.LoadImaged(keys=["volume","label"], reader="NibabelReader"),
            tfs.EnsureChannelFirstd(keys=["volume","label"]),
            tfs.Spacingd(keys=["volume","label"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            tfs.Resized(keys=["volume","label"], spatial_size=[128,128,128]),
            tfs.NormalizeIntensityd(keys=["volume"]),
        ])
        val_set = monai.data.Dataset(data=val_files, transform=val_transforms)

        # train_set = monai.data.Dataset(data=train_files, transform=train_transforms)
        train_loader =DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
        # val_set = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader =DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=list_data_collate,
        )
        return train_loader, val_loader
    elif mode == "test":
        base_transforms = tfs.Compose([
            tfs.LoadImaged(keys=["volume","label"], reader="NibabelReader"),
            tfs.EnsureChannelFirstd(keys=["volume","label"]),
            tfs.Spacingd(keys=["volume","label"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            tfs.Resized(keys=["volume","label"], spatial_size=[128,128,128]),
            tfs.NormalizeIntensityd(keys=["volume"]), 
        ])
        test_loader = monai.data.Dataset(data=base_files, transform=base_transforms)

        # train_set = monai.data.Dataset(data=train_files, transform=train_transforms)
        test_loader =DataLoader(
            test_loader,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=list_data_collate,
        )
        return test_loader
