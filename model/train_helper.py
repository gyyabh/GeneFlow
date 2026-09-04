# 训练工具 + 流匹配采样
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from tqdm import tqdm


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                             Flow Matching Sampling                            #
#################################################################################

@torch.no_grad()
def flow_sample(model, cond, num_steps=100, device="cuda", show_progress=False):
    """Sample x_0 from a trained flow-matching Stem model.

    Args:
        model: StemModel or DDP-wrapped StemModel (optionally EMA weights).
        cond: (B, cond_dim) condition embeddings.
        num_steps: number of integration steps from t=1 -> t=0.
        device: torch device string.

    Returns:
        x0: (B, NumGene) sampled gene expression.
    """

    # unwrap DDP if needed
    net = model.module if hasattr(model, "module") else model
    net.eval()

    cond = cond.to(device)
    B = cond.size(0)

    # assume StemModel has attribute input_size = NumGene
    num_genes = net.input_size
    x = torch.randn(B, num_genes, device=device)

    # time grid from 1.0 -> 0.0
    t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    iterator = tqdm(range(num_steps)) if show_progress else range(num_steps)
    for i in iterator:
        t = t_vals[i].expand(B)  # (B,)
        v = net(x, t, cond)      # (B, NumGene)
        dt = t_vals[i + 1] - t_vals[i]
        x = x + v * dt

    return x
