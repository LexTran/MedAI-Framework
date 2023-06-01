import SimpleITK as sitk
import os
import argparse

parser = argparse.ArgumentParser(description='Convert mhd to nii')
parser.add_argument('--dir', default=None, help='data path')
parser.add_argument('--output_dir', default=None, help='output path')
args = parser.parse_args()

dir = args.dir
out_dir = args.output_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for file in sorted(os.listdir(dir)):
    if file[-3:]=="mhd":
        file_name = dir + '/' + file
        prefix = file.split('.mhd')[0]
        save_name = out_dir + prefix + '.nii.gz'
        itkimage = sitk.ReadImage(file_name)
        
        spacing = itkimage.GetSpacing()

        out_arr = sitk.GetArrayFromImage(itkimage)
        out = sitk.GetImageFromArray(out_arr)
        out.SetSpacing(spacing)
        sitk.WriteImage(out, save_name)

print("end")