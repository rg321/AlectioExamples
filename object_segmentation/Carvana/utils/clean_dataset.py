import shutil
import os
from glob import glob

imgs_dir = "./data/imgs/train/training/"
masks_dir = "./data/masks/train_masks/training/"
file_ids = [os.path.splitext(file)[0] for file in os.listdir(
    imgs_dir) if not file.startswith(".")]
for ids in file_ids:
    mask_file = glob(masks_dir + ids + "_mask.*")
    img_file = glob(imgs_dir + ids + ".*")
    if len(img_file) != len(mask_file):
        print("Removing ", img_file)
        os.remove(img_file[0])
