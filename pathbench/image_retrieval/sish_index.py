import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader
import slideflow as sf
import logging 

from ..image_retrieval.utils import load_patch_dicts_pickle

logger = logging.getLogger(__name__)

class PatchTFRecordDataset(torch.utils.data.Dataset):
    def __init__(self, mosaic_pkl_path: str, transform):
        # load the mosaic “properties” + “patches”
        data = load_patch_dicts_pickle(mosaic_pkl_path, reconstruct_features=False)
        self.patches     = data["patches"]
        self.tfr_path    = data["properties"]["tfr_path"]
        self.transform   = transform
        self._tfr        = sf.TFRecord(self.tfr_path)
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        rec = self._tfr[self.patches[idx]["tfr_index"]]
        pil_img = sf.io.decode_image(bytes(rec["image_raw"]))  # PIL.Image

        # Convert PIL→numpy→Tensor, cast and scale
        arr = np.array(pil_img)                    # H×W×C, uint8 [0,255]
        tensor = torch.from_numpy(arr)             # ByteTensor [0,255]
        tensor = tensor.permute(2,0,1).float()     # FloatTensor [0,255]
        tensor = tensor.div(255.0)                 # FloatTensor [0,1]
        tensor = tensor.mul(2).sub(1)              # FloatTensor in [-1,1]

        return tensor
    
def compute_latent_features(
    mosaic_pkl: str,
    transform,
    vqvae: nn.Module,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4
) -> np.ndarray:
    ds = PatchTFRecordDataset(mosaic_pkl, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_latents = []
    vqvae.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            latents = vqvae(batch)       # (B, H, W) integer indices
            all_latents.append(latents.cpu().numpy())
            
    return np.concatenate(all_latents, 0)

def to_latent_semantic(latent, codebook_semantic):
    """
    Convert the original VQ-VAE latent code by using re-ordered codebook
    Input:
        latent (64 x 64 np.array): The original latent code from VQ-VAE encoder
        codebook_semantic (dict): The dictionary that map old codewords to
        the new ones
    Output:
        latent_semantic: The latent code with new codewords
    """
    latent_semantic = np.zeros_like(latent)
    for i in range(latent_semantic.shape[0]):
        for j in range(latent_semantic.shape[1]):
            latent_semantic[i][j] = codebook_semantic[latent[i][j]]
    return latent_semantic


def slide_to_index(latent, codebook_semantic, pool_layers, pool=None):
    """
    Convert VQ-VAE latent code into an integer
    Input:
        latent (N x 64 x 64 np array): The latent code from VQ-VAE enecoder
        codebook_semantic (128 x 256): The codebook from VQ-VAE encoder
        pool_layers (torch.nn.Sequential): A series of pool layers that convert the latent code into an integer
    Output:
        index (int): An integer index that represents the latent code
    """
    if pool is None:
        arr = np.asarray(latent)
        # single 64×64 map → make a batch of size 1
        if arr.ndim == 2:
            sem = to_latent_semantic(arr, codebook_semantic)        # (64,64)
            feat = torch.from_numpy(sem[np.newaxis, ...])          # (1,64,64)
        # already a batch of N maps
        elif arr.ndim == 3:
            sem_list = [to_latent_semantic(arr[i], codebook_semantic)
                        for i in range(arr.shape[0])]
            feat = torch.from_numpy(np.stack(sem_list, axis=0))    # (N,64,64)
        else:
            raise ValueError(f"Expected latent of ndim 2 or 3, got {arr.ndim}")
    else:
        iterable = [(lt, codebook_semantic) for lt in latent]
        result = pool.starmap(to_latent_semantic, iterable)
        feat = torch.from_numpy(np.array(result))

    num_level = list(range(len(pool_layers) + 1))
    level_sum_dict = {level: None for level in num_level}
    for level in num_level:
        if level == 0:
            level_sum_dict[level] = torch.sum(feat, (1, 2)).numpy().astype(float)
        else:
            feat = pool_layers[level - 1](feat)
            level_sum_dict[level] = torch.sum(feat, (1, 2)).numpy().astype(float)

    level_power = [0, 0, 1e6, 1e11]
    index = 0
    for level, power in enumerate(level_power):
        if level == 1:
            index = copy.deepcopy(level_sum_dict[level])
        elif level > 1:
            index += level_sum_dict[level] * power
            
    return index

def min_max_binarized(feat):
    """
    Min-max algorithm proposed in paper: Yottixel-An Image Search Engine for Large Archives of
    Histopathology Whole Slide Images.
    Input:
        feat (1 x 1024 np.arrya): Features from the last layer of DenseNet121.
    Output:
        output_binarized (str): A binary code of length  1024
    """
    prev = float('inf')
    output_binarized = []
    for ele in feat:
        if ele < prev:
            code = 0
            output_binarized.append(code)
        elif ele >= prev:
            code = 1
            output_binarized.append(code)
        prev = ele
    output_binarized = "".join([str(e) for e in output_binarized])
    return output_binarized
