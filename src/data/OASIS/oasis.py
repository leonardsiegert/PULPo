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

def convert_to_onehot(
    labels: torch.Tensor, num_classes: int, channel_dim: int = 1
) -> torch.Tensor:

    out = F.one_hot(labels.long(), num_classes)
    out = out.unsqueeze(channel_dim).transpose(channel_dim, out.dim())
    return out.squeeze(-1)

def create_data_loaders(batch_size, segs=False, lms=False, mask=False, ndims=3):
    if mask == True:
        raise NotImplementedError("Mask not implemented for OASIS")
    if lms == True:
        print("CAREFUL: Landmarks for OASIS exists only for the test_lm split")

    train_set = OASIS(split="training", segs=segs, lms=False, mask=False, ndims=ndims)
    val_set = OASIS(split="validation", segs=segs, lms=False, mask=False, ndims=ndims)
    test_set_seg = OASIS(split="test_seg", segs=segs, lms=False, mask=False, ndims=ndims)
    test_set_lm = OASIS(split="test_lm", segs=False, lms=lms, mask=False, ndims=ndims)

    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size, num_workers = 1, drop_last=False)
    validation_loader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=batch_size, num_workers = 1, drop_last=False)
    test_loader_seg = DataLoader(test_set_seg, sampler=SequentialSampler(test_set_seg), batch_size=1, num_workers = 1, drop_last=False)
    test_loader_lm = DataLoader(test_set_lm, sampler=SequentialSampler(test_set_lm), batch_size=1, num_workers = 1, drop_last=False)
    print(
        "Number of training/validation/test_seg/test_lm patches:",
        (train_set.__len__(),val_set.__len__(), test_set_seg.__len__(), test_set_lm.__len__()),
    )

    return train_loader, validation_loader, test_loader_seg, test_loader_lm


class OASIS(Dataset):
    def __init__(self, split, segs=False, lms=False, mask=False, ndims=3):
        self.path = pathlib.Path(__file__).parent.resolve()
        self.segs = segs
        self.lms = lms
        self.mask = mask
        if self.mask == True:
            raise NotImplementedError("Mask not implemented for OASIS")
        self.split = split
        self.ndims=ndims
        with h5py.File(os.path.join(self.path, "OASIS.h5"), "r") as f:
            self.input_size = f.attrs["shape"]
            self.length = f[self.split].attrs["N"]
        
    def __getitem__(self, index):
        index2 = index
        # this loop makes sure that the two images are not the same
        while index2 == index:
            index2 = random.randint(0, self.__len__() - 1)

        with h5py.File(os.path.join(self.path, "OASIS.h5"), "r") as f:
            img1 = f[self.split]['image'][str(index)][:,:,:]
            img2 = f[self.split]['image'][str(index2)][:,:,:]
            img1 = torch.from_numpy(img1).type(torch.float32).unsqueeze(0)
            img2 = torch.from_numpy(img2).type(torch.float32).unsqueeze(0)
            if self.segs:
                seg1 = f[self.split]['seg'][str(index)][:]
                seg2 = f[self.split]['seg'][str(index2)][:]
                seg1 =  torch.from_numpy(seg1).type(torch.float32)
                seg2 = torch.from_numpy(seg2).type(torch.float32)
                seg1 = convert_to_onehot(seg1, num_classes=f[self.split].attrs["seg_dim"], channel_dim=0).type(torch.float32)
                seg2 = convert_to_onehot(seg2, num_classes=f[self.split].attrs["seg_dim"], channel_dim=0).type(torch.float32)
            else:
                seg1 = torch.empty((0,), dtype=torch.float32)
                seg2 = torch.empty((0,), dtype=torch.float32)
            if self.lms:
                lms1 = f[self.split]['landmarks'][str(index)][:]
                lms2 = f[self.split]['landmarks'][str(index2)][:]
                lms1 = torch.from_numpy(lms1).type(torch.float32)
                lms2 = torch.from_numpy(lms2).type(torch.float32)
            else:
                lms1 = torch.empty((0,), dtype=torch.float32)
                lms2 = torch.empty((0,), dtype=torch.float32)

        mask1 = torch.empty((0,), dtype=torch.float32)
        mask2 = torch.empty((0,), dtype=torch.float32)    
            
        return img1, img2, seg1, seg2, lms1, lms2, mask1, mask2

            
    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.length