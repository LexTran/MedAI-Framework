from glob import glob
import os
import math
import numpy as np

import torch
from monai.data import DataLoader,list_data_collate,decollate_batch,pad_list_data_collate
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
    if mode == "train":
        length = len(base_files) 
        len_list = [math.floor(0.9*length), math.ceil(0.1*length)]
        train_files, val_files = torch.utils.data.random_split(base_files,lengths=len_list,generator=torch.Generator().manual_seed(42))
    elif mode == "test":
        test_files = base_files

    # how to load the data
    if mode == "train":
        train_transforms = tfs.Compose([
            tfs.LoadImaged(keys=["volume","label"]),
            tfs.EnsureChannelFirstd(keys=["volume","label"]),
            tfs.NormalizeIntensityd(keys=["volume"]),
            tfs.CropForegroundd(keys=["volume", "label"], source_key="volume"),
            tfs.Orientationd(keys=["volume", "label"], axcodes="RAS"),
            tfs.Spacingd(keys=["volume","label"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            tfs.EnsureTyped(keys=["volume", "label"], device=torch.device("cpu"), track_meta=False),
            tfs.RandCropByPosNegLabeld(
                keys=["volume", "label"],
                label_key="label",
                spatial_size=(48, 48, 48),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="volume",
                image_threshold=0,
            ),
            tfs.RandRotated(keys=["volume","label"],range_x=[0.4,0.4],range_y=[0.4,0.4],range_z=[0.4,0.4],prob=0.5,
                            keep_size=True,mode="bilinear",padding_mode="zeros"),
            tfs.RandFlipd(keys=["volume","label"],spatial_axis=0,prob=0.1),
            tfs.RandFlipd(keys=["volume","label"],spatial_axis=1,prob=0.1),
            tfs.RandFlipd(keys=["volume","label"],spatial_axis=2,prob=0.1),
        ])
        train_set = monai.data.CacheDataset(data=train_files,
                                            transform=train_transforms,
                                            cache_num=24,
                                            cache_rate=1.0,
                                            num_workers=8,)
        import pdb
        pdb.set_trace()
        val_transforms = tfs.Compose([
            tfs.LoadImaged(keys=["volume","label"]),
            tfs.EnsureChannelFirstd(keys=["volume","label"]),
            tfs.NormalizeIntensityd(keys=["volume"]),
            tfs.CropForegroundd(keys=["volume", "label"], source_key="volume"),
            tfs.Orientationd(keys=["volume", "label"], axcodes="RAS"),
            tfs.Spacingd(keys=["volume","label"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            tfs.EnsureTyped(keys=["volume", "label"], device=torch.device("cpu"), track_meta=True),
        ])
        val_set = monai.data.CacheDataset(data=val_files,
                                            transform=val_transforms,
                                            cache_num=6,
                                            cache_rate=1.0,
                                            num_workers=4,)

        # train_set = monai.data.Dataset(data=train_files, transform=train_transforms)
        train_loader =DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=pad_list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
        # val_set = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader =DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=pad_list_data_collate,
        )
        return train_loader, val_loader
    elif mode == "test":
        test_transforms = tfs.Compose([
            tfs.LoadImaged(keys=["volume","label"]),
            tfs.EnsureChannelFirstd(keys=["volume","label"]),
            tfs.NormalizeIntensityd(keys=["volume"]),
            tfs.CropForegroundd(keys=["volume", "label"], source_key="volume"),
            tfs.Orientationd(keys=["volume", "label"], axcodes="RAS"),
            tfs.Spacingd(keys=["volume","label"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            tfs.EnsureTyped(keys=["image", "label"], device=torch.device("cpu"), track_meta=True),
        ])
        test_loader = monai.data.CacheDataset(data=test_files,
                                            transform=test_transforms,
                                            cache_num=6,
                                            cache_rate=1.0,
                                            num_workers=4,)

        # train_set = monai.data.Dataset(data=train_files, transform=train_transforms)
        test_loader =DataLoader(
            test_loader,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=pad_list_data_collate,
        )
        return test_loader
