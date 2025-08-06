import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pathlib
import json
import random
import pickle
import nibabel
import h5py
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import itertools
import torch.nn.functional as F


def create_data_loaders(batch_size, segs=False, lms=False, mask=False, ndims=3, interpatient=False):

    if interpatient:
        train_set = BraTS_interpatient(split="training", segs=False, lms=lms, mask=mask, ndims=ndims)
        val_set = BraTS_interpatient(split="validation", segs=False, lms=False, mask=mask, ndims=ndims)
        test_set = BraTS_interpatient(split="test", segs=False, lms=lms, mask=mask, ndims=ndims)
    else:
        train_set = BraTS(split="training", segs=False, lms=lms, mask=mask, ndims=ndims)
        val_set = BraTS(split="validation", segs=False, lms=False, mask=mask, ndims=ndims)
        test_set = BraTS(split="test", segs=False, lms=lms, mask=mask, ndims=ndims)

    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size, num_workers = 1, drop_last=False)
    validation_loader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=batch_size, num_workers = 1, drop_last=False)
    test_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=batch_size, num_workers = 1, drop_last=False)
    print(
        "Number of training/validation patches:",
        (train_set.__len__(),val_set.__len__(), test_set.__len__())
    )

    return train_loader, validation_loader, test_loader


class BraTS(Dataset):
    def __init__(self, split, segs=False, lms=False, mask=False, ndims=3):
        self.path = pathlib.Path(__file__).parent.resolve()
        self.segs = segs
        self.lms = lms
        self.mask = mask
        if self.segs:
            raise ValueError("Segs not implemented")
        if self.mask:
            raise ValueError("Mask not implemented")
        with h5py.File(os.path.join(self.path, "BraTS.h5"), "r") as f:
            self.input_size = f.attrs["shape"]
            self.input_size[-1] = self.input_size[-1]
            print("input_size", self.input_size)
            self.split = split # training, validation
            self.ndims=ndims
            self.length = f[self.split].attrs["N"]

    def __getitem__(self, index):
        with h5py.File(os.path.join(self.path, "BraTS.h5"), "r") as f:

            if self.ndims == 2:
                raise ValueError("2D not implemented")
            else: # 3D
                follow_img = f[self.split]["follow"]["t1ce"][str(index)][:,:,:]
                follow_img = torch.from_numpy(follow_img).type(torch.float32).unsqueeze(0)
                base_img = f[self.split]["base"]["t1ce"][str(index)][:,:,:]
                base_img = torch.from_numpy(base_img).type(torch.float32).unsqueeze(0)
                
                if self.lms:
                    follow_lms = f[self.split]["follow"]["landmarks"][str(index)][:]
                    follow_lms = torch.from_numpy(follow_lms).type(torch.float32)
                    if self.split == "validation":
                        base_lms = torch.empty((0,), dtype=torch.float32)
                    else:
                        base_lms = f[self.split]["base"]["landmarks"][str(index)][:]
                        base_lms = torch.from_numpy(base_lms).type(torch.float32)
                else:
                    follow_lms = torch.empty((0,), dtype=torch.float32)
                    base_lms = torch.empty((0,), dtype=torch.float32)

                follow_seg = torch.empty((0,), dtype=torch.float32)
                base_seg = torch.empty((0,), dtype=torch.float32)
                follow_mask = torch.empty((0,), dtype=torch.float32)
                base_mask = torch.empty((0,), dtype=torch.float32)
                
            return follow_img, base_img, follow_seg, base_seg, follow_lms, base_lms, follow_mask, base_mask

    # Override to give PyTorch size of dataset
    def __len__(self):
            return self.length

class BraTS_interpatient(Dataset):
    def __init__(self, split, segs=False, lms=False, mask=False, ndims=3):
        self.path = pathlib.Path(__file__).parent.resolve()
        self.segs = segs
        self.lms = lms
        self.mask = mask
        if self.segs:
            raise ValueError("Segs not implemented")
        if self.mask:
            raise ValueError("Mask not implemented")
        if self.lms:
            print("Landmarks don't work with interpatient. Different number of landmarks for each patient.")
        with h5py.File(os.path.join(self.path, "BraTS.h5"), "r") as f:
            self.input_size = f.attrs["shape"]
            print("input_size", self.input_size)
            self.split = split # training, validation
            self.ndims=ndims
            self.length = f[self.split].attrs["N"]

    def __getitem__(self, index):
        with h5py.File(os.path.join(self.path, "BraTS.h5"), "r") as f:

            if self.ndims == 2:
                raise ValueError("2D not implemented")
            else: # 3D
                index1 = index
                # generate random binary
                coin1 = "follow" if random.randint(0,1) == 0 else "base"
                coin2 = "follow" if random.randint(0,1) == 0 else "base"
                # generate random second index
                index2 = random.randint(0, self.length-1)
                while index2 == index and coin1 == coin2:
                    index2 = random.randint(0, self.length-1)

                print("index1", index1, "coin1", coin1, "index2", index2, "coin2", coin2)

                moving = f[self.split][coin1]["t1ce"][str(index1)][:,:,:]
                moving = torch.from_numpy(moving).type(torch.float32).unsqueeze(0)
                fixed = f[self.split][coin2]["t1ce"][str(index2)][:,:,:]
                fixed = torch.from_numpy(fixed).type(torch.float32).unsqueeze(0)
                
                if self.lms:
                    moving_lms = f[self.split][coin1]["landmarks"][str(index1)][:]
                    moving_lms = torch.from_numpy(moving_lms).type(torch.float32)
                    if self.split == "validation":
                        moving_lms = torch.empty((0,), dtype=torch.float32)
                    else:
                        fixed_lms = f[self.split][coin2]["landmarks"][str(index2)][:]
                        fixed_lms = torch.from_numpy(fixed_lms).type(torch.float32)
                else:
                    moving_lms = torch.empty((0,), dtype=torch.float32)
                    fixed_lms = torch.empty((0,), dtype=torch.float32)

                moving_seg = torch.empty((0,), dtype=torch.float32)
                fixed_seg = torch.empty((0,), dtype=torch.float32)
                moving_mask = torch.empty((0,), dtype=torch.float32)
                fixed_mask = torch.empty((0,), dtype=torch.float32)
                
            return moving, fixed, moving_seg, fixed_seg, moving_lms, fixed_lms, moving_mask, fixed_mask

    # Override to give PyTorch size of dataset
    def __len__(self):
            return self.length
