import SimpleITK as sitk
import numpy as np
import mcubes
import os
from skimage import measure
from stl import mesh
import argparse

parser = argparse.ArgumentParser(description='Convert mhd to nii')
parser.add_argument('--dir', default=None, help='data path')
parser.add_argument('--output_dir', default=None, help='output path')
args = parser.parse_args()

# Load NIfTI file
for data in sorted(os.listdir(args.dir)):
    vol = sitk.ReadImage(args.dir+"/"+data)
    vol_array = sitk.GetArrayFromImage(vol)

    # Threshold the data (optional)
    threshold = -500  # Adjust this value based on your needs
    binary_data = np.where(vol_array >= threshold, 1, 0)

    # Generate surface mesh using marching cubes algorithm
    vertices, faces, _, _ = measure.marching_cubes(binary_data,method='lewiner')

    # Create STL mesh object
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[face[j], :]

    stl_file = "original.stl"
    stl_mesh.save(args.output_dir+"/"+stl_file)
