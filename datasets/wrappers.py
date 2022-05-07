import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples

import ants

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr, d_lr = img_lr.shape
            # img_hr = img_hr[:h_lr * s, :w_lr * s, :d_lr*s]
            img_hr = ants.crop_indices(img_hr, (0,0,0), (h_lr * s,w_lr * s,d_lr*s))
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[0] - w_lr)
            y0 = random.randint(0, img_lr.shape[1] - w_lr)
            z0 = random.randint(0, img_lr.shape[2] - w_lr)
            crop_lr = ants.crop_indices(img_lr, (x0,y0,z0),(x0 + w_lr, y0 + w_lr, z0 + w_lr))
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            z1 = z0 * s
            crop_hr = ants.crop_indices(img_hr, (x1, y1, z1), (x1 + w_hr, y1 + w_hr, z1 + w_hr))

        crop_hr = to_tensor(crop_hr)
        crop_lr = to_tensor(crop_lr)

        if self.augment:
            xflip = random.random() < 0.5
            yflip = random.random() < 0.5
            zflip = random.random() < 0.5

            xtranspose = random.random() < 0.5
            ytranspose = random.random() < 0.5
            ztranspose = random.random() < 0.5

            def augment(x):
                if xflip:
                    x = x.flip(1)
                if yflip:
                    x = x.flip(2)
                if zflip:
                    x = x.flip(3)
                if xtranspose:
                    x = x.transpose(2, 3)
                if ytranspose:
                    x = x.transpose(1, 3)
                if ztranspose:
                    x = x.transpose(1, 2)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[1]
        cell[:, 1] *= 2 / crop_hr.shape[2]
        cell[:, 2] *= 2 / crop_hr.shape[3]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


def to_tensor(img):
    # img=(img-img.min())/(img.max()-img.min()+1e-10)
    return torch.Tensor(img.numpy(True).transpose(3, 0, 1, 2))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[0] / s + 1e-9)
            w_lr = math.floor(img.shape[1] / s + 1e-9)
            d_lr = math.floor(img.shape[2] / s + 1e-9)
            img = ants.crop_indices(img, (0,0,0), (round(h_lr * s),round(w_lr * s),round(d_lr * s)))
            img_down = ants.resample_image(img, (h_lr, w_lr, d_lr), use_voxels=True, interp_type=4)
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[0] - w_hr)
            y0 = random.randint(0, img.shape[1] - w_hr)
            z0 = random.randint(0, img.shape[2] - w_hr)
            crop_hr = ants.crop_indices(img, (x0,y0,z0), (x0 + w_hr,y0 + w_hr,z0 + w_hr))
            crop_lr = ants.resample_image(crop_hr, (w_lr, w_lr, w_lr), use_voxels=True, interp_type=4)

        crop_hr = to_tensor(crop_hr)
        crop_lr = to_tensor(crop_lr)

        if self.augment:
            xflip = random.random() < 0.5
            yflip = random.random() < 0.5
            zflip = random.random() < 0.5

            xtranspose = random.random() < 0.5
            ytranspose = random.random() < 0.5
            ztranspose = random.random() < 0.5

            def augment(x):
                if xflip:
                    x = x.flip(1)
                if yflip:
                    x = x.flip(2)
                if zflip:
                    x = x.flip(3)
                if xtranspose:
                    x = x.transpose(2, 3)
                if ytranspose:
                    x = x.transpose(1, 3)
                if ztranspose:
                    x = x.transpose(1, 2)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[1]
        cell[:, 1] *= 2 / crop_hr.shape[2]
        cell[:, 2] *= 2 / crop_hr.shape[3]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = ants.resample_image(img_hr, (w_hr, w_hr, w_hr), use_voxels=True, interp_type=4)
        if self.gt_resize is not None:
            img_hr = ants.resample_image(img_hr, (self.gt_resize, self.gt_resize, self.gt_resize), use_voxels=True, interp_type=4)

        img_hr = to_tensor(img_hr)
        img_lr = to_tensor(img_lr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[1]
        cell[:, 1] *= 2 / img_hr.shape[2]
        cell[:, 2] *= 2 / img_hr.shape[3]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
