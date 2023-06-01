import SimpleITK as sitk
import os
import argparse

parser = argparse.ArgumentParser(description='Convert mhd to nii')
parser.add_argument('--dir', default=None, help='data path')
parser.add_argument('--output_dir', default=None, help='output path')
args = parser.parse_args()

file_dir = args.dir
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

list_name = []
output_name = []

for series in os.listdir(file_dir):
    sub_dir1 = False
    sub_dir2 = False
    if os.path.isdir(file_dir+series):
        for root, dirs, file in os.walk(file_dir + series):
            if dirs: 
                for root, dirs, file in os.walk(root + "/" + dirs[0]):
                    if dirs:
                        root = root + "/" + dirs[0]
                        list_name.append(root)
                        output_name.append((root.split("\\")[0]).split("/")[3])
                        sub_dir2 = True
                        break
                    else:
                        if file[0].endswith(".ima") or file[0].endswith(".IMA") or file[0].endswith(".dcm"):
                            list_name.append(root)
                            output_name.append((root.split("\\")[0]).split("/")[3])
                            sub_dir1 = True
                            break
                if sub_dir1 or sub_dir2:
                    break
            else:
                if file[0].endswith(".ima") or file[0].endswith(".IMA") or file[0].endswith(".dcm"):
                    list_name.append(root)
                    output_name.append(root.split("/")[2])
                    break

for i in range(len(list_name)):
    try:
        if os.path.isfile(output_dir + output_name[i] + ".nii.gz"):
            pass
        else:
            file_path = list_name[i]
            series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path, series_id[0])
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(series_file_names)
            images = series_reader.Execute()
            sitk.WriteImage(images, output_dir + output_name[i] + ".nii.gz")
    except:
        print(output_name[i])