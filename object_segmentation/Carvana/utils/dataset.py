from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.ids = []
        for file_name in listdir(imgs_dir):
            if not file_name.startswith("."):
                idx = splitext(file_name)[0]
                mask_file = glob(masks_dir + idx + "_mask.*")
                img_file = glob(imgs_dir + idx + ".*")
                if len(img_file) != len(mask_file):
                    print("Removing files without masks ", img_file[0])
                    os.remove(img_file[0])
                else:
                    self.ids.append(idx)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + "_mask.*")
        img_file = glob(self.imgs_dir + idx + ".*")
        assert (
            len(mask_file) == 1
        ), "Either no mask or multiple masks found for the ID {}: {}".format(idx, mask_file)
        assert (
            len(img_file) == 1
        ), "Either no image or multiple images found for the ID {}: {}".format(idx, mask_file)
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert (
            img.size == mask.size
        ), "Image and mask {} should be the same size, but are {} and {}".format(idx, img.size, mask.size)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            "image": torch.from_numpy(img).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask).type(torch.FloatTensor),
        }
