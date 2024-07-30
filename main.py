import os
from utils import uncompress_tar, load_nifti_image, create_gif_from_slices

# 解压缩数据集
uncompress_tar('/data/brats-2021-task1/BraTS2021_00495.tar', '/kaggle/working/brats-2021-task1/BraTS2021_00495')
uncompress_tar('/data//brats-2021-task1/BraTS2021_00621.tar', '/kaggle/working/brats-2021-task1/BraTS2021_00621')
uncompress_tar('/data/brats-2021-task1/BraTS2021_Training_Data.tar', '/kaggle/working/brats-2021-task1/BraTS2021_Training_Data')

# 加载和显示NIfTI图像数据
nifti_t1_path = '/working/brats-2021-task1/BraTS2021_Training_Data/BraTS2021_00577/BraTS2021_00577_t1.nii.gz'
img_t1 = load_nifti_image(nifti_t1_path)
create_gif_from_slices(img_t1, axis=2, duration=0.1, resize_to=(100, 100))

nifti_t1ce_path = '/working/brats-2021-task1/BraTS2021_Training_Data/BraTS2021_00577/BraTS2021_00577_t1ce.nii.gz'
img_t1ce = load_nifti_image(nifti_t1ce_path)
create_gif_from_slices(img_t1ce, axis=2, duration=0.1, resize_to=(100, 100))

nifti_t2_path = '/working/brats-2021-task1/BraTS2021_Training_Data/BraTS2021_00577/BraTS2021_00577_t2.nii.gz'
img_t2 = load_nifti_image(nifti_t2_path)
create_gif_from_slices(img_t2, axis=2, duration=0.1, resize_to=(100, 100))

nifti_flair_path = '/working/brats-2021-task1/BraTS2021_Training_Data/BraTS2021_00577/BraTS2021_00577_flair.nii.gz'
img_flair = load_nifti_image(nifti_flair_path)
create_gif_from_slices(img_flair, axis=2, duration=0.1, resize_to=(100, 100))

nifti_seg_path = '/working/brats-2021-task1/BraTS2021_Training_Data/BraTS2021_00577/BraTS2021_00577_seg.nii.gz'
img_seg = load_nifti_image(nifti_seg_path)
create_gif_from_slices(img_seg, axis=2, duration=0.1, resize_to=(100, 100))
