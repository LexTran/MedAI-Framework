import torch
import numpy as np
import os
import SimpleITK as sitk
import argparse

parser = argparse.ArgumentParser(description='Medical 3D Segmentation')
parser.add_argument('--data_root', default=None, help="where you put your data")
args = parser.parse_args()

# load multiple labels
# --label
# ----heart
# ------label1.nii.gz
# ----lung
# ------label1.nii.gz

label_root = args.data_root
sub_dir = sorted(os.listdir(label_root))
num_class = len(sub_dir)
label_path = sorted([label_root+"/"+i for i in sub_dir])

for label in sorted(os.listdir(label_path)):
    name = label.split(".nii.gz")[0]

    for idx in range(num_class):
        organ = sitk.ReadImage(label_path+"/"+label)
        organ = sitk.GetArrayFromImage(organ)
        if organ_combine is None:
            organ_combine = np.zeros(organ.shape)
        organ[organ>0] = idx+1
        organ_combine += organ

    organ_combine = sitk.GetImageFromArray(organ_combine)
    organ_combine.SetDirection([1,0,0,0,1,0,0,0,1])
    organ_combine.SetOrigin([0,0,0])
    organ_combine.SetSpacing([1,1,1])
    if not os.path.exists(label_path+"/combined"):
        os.mkdir(label_path+"/combined")
    sitk.WriteImage(organ_combine, label_path+"/combined/"+name+".nii.gz")


