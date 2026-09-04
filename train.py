# 完整 GeneFlow 训练脚本
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
import time
import argparse
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm
import random
import anndata

import sys
sys.path.append("./Stem")
from Stem.models import Stem_models
from Stem.train_helper import *


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def ddp_setup(rank, world_size, available_gpus):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(available_gpus[rank])


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        rank: int,
        gpu_id: int,
        model_args: argparse.Namespace,
    ) -> None:
        self.rank = rank
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.args = model_args
        
        self.model = model
        self.ema = deepcopy(model).to(gpu_id)
        requires_grad(self.ema, False)
        self.model = DDP(self.model.to(gpu_id), device_ids=[self.gpu_id], find_unused_parameters=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=self.args.lr, weight_decay=0)
        update_ema(self.ema, self.model.module, decay=0)
        self.args.logger.info(f"Rank {rank} - Initializing Trainer... DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

        self.train_steps=0
        self.log_steps=0
        self.running_loss=0

    def _run_batch(self, x0, cond):
        """Run one Flow Matching batch.

        Args:
            x0: (B, NumGene) clean gene expression.
            cond: (B, cond_dim) image embeddings.
        """
        x0 = x0.to(self.gpu_id)
        cond = cond.to(self.gpu_id)
        B = x0.size(0)

        path_type = getattr(self.args, "path_type", "ot")

        if path_type == "ot":
            # ===== 原 OT-CFM =====
            t = torch.rand(B, device=self.gpu_id)
            eps = torch.randn_like(x0)
            t_view = t.view(B, 1)
            x_t = (1.0 - t_view) * x0 + t_view * eps
            v_true = eps - x0

        elif path_type == "i":
            # ===== I-CFM: 时间离散 / 固定 =====
            # 方案 A：从离散集合 {0.25, 0.5, 0.75} 中采样
            candidate_ts = torch.tensor([0.25, 0.5, 0.75], device=self.gpu_id)
            idx = torch.randint(low=0, high=candidate_ts.numel(), size=(B,), device=self.gpu_id)
            t = candidate_ts[idx]

            # 也可以改成固定 t=0.5：
            # t = torch.full((B,), 0.5, device=self.gpu_id)

            eps = torch.randn_like(x0)
            t_view = t.view(B, 1)
            x_t = (1.0 - t_view) * x0 + t_view * eps
            v_true = eps - x0

        elif path_type == "gauss":
            # 消融路径：高斯 -> 数据
            t = torch.rand(B, device=self.gpu_id)      # t ~ U(0,1)
            z = torch.randn_like(x0)                   # z ~ N(0, I)
            t_view = t.view(B, 1)
            x_t = z + t_view * (x0 - z)               # x_t = z + t (x0 - z)
            v_true = x0 - z

        else:
            raise ValueError(f"Unknown path_type: {path_type}")

        # ======================================================================
        # Modified: Multi-Scale Loss Calculation (Memory Efficient)
        # ======================================================================
        
        # Zero gradients first
        self.optimizer.zero_grad()
        
        # 1. Main Loss: Combined scales
        v_pred_total = self.model(x_t, t, cond, use_scale_idx=None)
        loss_total = torch.mean((v_pred_total - v_true) ** 2)
        loss_total.backward() # Backprop immediately to free graph
        
        # Keep track for logging
        total_loss_val = loss_total.item()

        # 2. Auxiliary Losses: Individual scale predictions
        lambda_aux = 1.0 
        
        # Scales: 0=224, 1=112, 2=56
        for i in range(3):
            # Re-run forward pass for this scale
            # Note: We don't need to retain graph from previous passes
            v_pred_scale = self.model(x_t, t, cond, use_scale_idx=i)
            loss_scale = torch.mean((v_pred_scale - v_true) ** 2)
            
            # Weighted backward
            (loss_scale * lambda_aux).backward()
            
            total_loss_val += (loss_scale.item() * lambda_aux)
        
        # Optimizer step
        self.optimizer.step()
        update_ema(self.ema, self.model.module)

        self.running_loss += total_loss_val
        self.train_steps += 1
        self.log_steps += 1
        if self.log_steps % 500 == 0:
            torch.cuda.synchronize()
            avg_loss = torch.tensor(self.running_loss / self.log_steps, device=self.gpu_id)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            self.args.logger.info(f"Step={self.train_steps:07d} | Training Loss: {avg_loss:.5f}")
            self.running_loss = 0
            self.log_steps = 0

        if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
            if self.rank == 0:
                self._save_checkpoint()
            dist.barrier()    


    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for x, y in self.train_data:
            # x: (B, NumGene), y: (B, cond_dim)
            self._run_batch(x, y)

    def _save_checkpoint(self):
        checkpoint = {
                      "model": self.model.module.state_dict(),
                      "ema": self.ema.state_dict(),
                      "opt": self.optimizer.state_dict()
                    }
        checkpoint_path = f"{self.args.checkpoint_dir}/{self.train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.args.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self, max_epochs: int):
        ##
        self.model.train()
        self.ema.eval()
        ##
        for epoch in range(max_epochs):
            self._run_epoch(epoch)


def assemble_dataset(input_args):
    # load & assemble data
    # leave the test slide out
    slidename_lst = list(np.genfromtxt(input_args.data_path + "processed_data/" + input_args.folder_list_filename, dtype=str))
    for slide_out in input_args.slide_out.split(","):
        slidename_lst.remove(slide_out)
        input_args.logger.info(f"{slide_out} is held out for testing.")
    input_args.logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")

    # load selected gene list
    selected_genes = list(np.genfromtxt(input_args.data_path + "processed_data/" + input_args.gene_list_filename, dtype=str))
    input_args.input_gene_size = len(selected_genes)
    input_args.logger.info(f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")

    resolutions = [224, 112, 56]
    def get_suffix(res):
        return "" if res==224 else f"_{res}"

    # load original patches
    first_slide = True
    all_img_ebd_ori = None
    all_count_mtx_ori = None
    input_args.logger.info("Loading original data...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]
        test_adata = anndata.read_h5ad(input_args.data_path + "st/" + sample_name + ".h5ad")
        test_count_mtx = pd.DataFrame(test_adata[:, selected_genes].X.toarray(), 
                                      columns=selected_genes, 
                                      index=[sample_name + "_" + str(i) for i in range(test_adata.shape[0])])
        
        # 1. load multi-resolution embeddings
        img_ebd_list = []
        for res in resolutions:
            suffix = get_suffix(res)
            img_ebd_uni   = torch.load(input_args.data_path + f"processed_data/1spot_uni_ebd/{sample_name}_uni{suffix}.pt", map_location="cpu")
            img_ebd_conch = torch.load(input_args.data_path + f"processed_data/1spot_conch_ebd/{sample_name}_conch{suffix}.pt", map_location="cpu")
            img_ebd_list.extend([img_ebd_uni, img_ebd_conch])
    
        slide_img_ebd = torch.cat(img_ebd_list, axis=1)

        if first_slide:
            all_count_mtx_ori = test_count_mtx
            all_img_ebd_ori = slide_img_ebd
            first_slide = False
        else:
            all_count_mtx_ori = np.concatenate((all_count_mtx_ori, test_count_mtx), axis=0)
            all_img_ebd_ori = torch.cat([all_img_ebd_ori, slide_img_ebd], axis=0)

        input_args.logger.info(f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape} | img ebd shape: {all_img_ebd_ori.shape}")
    input_args.cond_size = all_img_ebd_ori.shape[1]
    
    # load augmented patches
    first_slide = True
    all_img_ebd_aug = None
    input_args.logger.info(f"Augmentation data loading...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]

        img_ebd_aug_list = []
        for res in resolutions:
            suffix = get_suffix(res)
            img_ebd_uni   = torch.load(input_args.data_path + f"processed_data/1spot_uni_ebd_aug/{sample_name}_uni_aug{suffix}.pt", map_location="cpu")
            img_ebd_conch = torch.load(input_args.data_path + f"processed_data/1spot_conch_ebd_aug/{sample_name}_conch_aug{suffix}.pt", map_location="cpu")
            img_ebd_aug_list.append(torch.cat([img_ebd_uni, img_ebd_conch], axis=-1))

        slide_img_ebd_aug = torch.cat(img_ebd_aug_list, axis=-1)

        if first_slide:
            all_img_ebd_aug = slide_img_ebd_aug
            first_slide = False
        else:
            all_img_ebd_aug = torch.cat([all_img_ebd_aug, slide_img_ebd_aug], axis=0)

        input_args.logger.info(f"With augmentation {sample_name} loaded, slide_img_ebd shape: {slide_img_ebd_aug.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}")

     
    # randomly select augmented patches according to the input augmentation ratio (int)
    num_aug_ratio = input_args.num_aug_ratio
    all_count_mtx_aug = np.repeat(np.copy(all_count_mtx_ori), num_aug_ratio, axis=0)             # generate count matrix for all augmented patches
    selected_img_ebd_aug = torch.zeros((all_count_mtx_aug.shape[0], all_img_ebd_aug.shape[2]))
    for i in range(all_img_ebd_aug.shape[0]):                                                    # randomly select augmented patches
        selected_transpose_idx = np.random.choice(all_img_ebd_aug.shape[1], num_aug_ratio, replace=False)
        selected_img_ebd_aug[i*num_aug_ratio:(i+1)*num_aug_ratio, :] = all_img_ebd_aug[i, selected_transpose_idx, :]

    all_img_ebd = torch.cat([all_img_ebd_ori, selected_img_ebd_aug], axis=0)
    all_count_mtx = np.concatenate((all_count_mtx_ori, all_count_mtx_aug), axis=0)
    input_args.logger.info(f"{num_aug_ratio}:1 augmentation. CONCH+UNI. final count_mtx shape: {all_count_mtx.shape} | final img_ebd shape: {all_img_ebd.shape}")
    
    ################################################
    all_count_mtx_df = pd.DataFrame(all_count_mtx, columns=selected_genes, index=list(range(all_count_mtx.shape[0])))
    # remove the spot with all NAN/zero in count mtx
    all_count_mtx_all_nan_spot_index = all_count_mtx_df.index[all_count_mtx_df.isnull().all(axis=1)]
    all_count_mtx_all_zero_spot_index = all_count_mtx_df.index[all_count_mtx_df.sum(axis=1) == 0]
    input_args.logger.info(f"All NAN spot index: {all_count_mtx_all_nan_spot_index}")
    input_args.logger.info(f"All zero spot index: {all_count_mtx_all_zero_spot_index}")
    spot_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
    spot_idx_to_keep = list(set(all_count_mtx_df.index) - set(spot_idx_to_remove))
    all_count_mtx = all_count_mtx_df.loc[spot_idx_to_keep, :]
    all_img_ebd = all_img_ebd[spot_idx_to_keep, :]
    input_args.logger.info(f"After exclude rows with all nan/zeros: {all_count_mtx.shape}, {all_img_ebd.shape}")
    # only normalized by log2(+1)
    all_count_mtx_selected_genes = np.log2(all_count_mtx.loc[:, selected_genes] + 1).copy()
    input_args.logger.info(f"Selected genes count matrix shape: {all_count_mtx_selected_genes.shape}" )
    all_img_ebd.requires_grad_(False)
    alldataset = CustomDataset(torch.from_numpy(all_count_mtx_selected_genes.values).float(), 
                               all_img_ebd.float())    
    return alldataset, input_args


def load_train_objs(args):
    train_set, args = assemble_dataset(args)
    model = Stem_models[args.model](
        input_size=args.input_gene_size,
        depth= args.DiT_num_blocks,
        hidden_size=args.hidden_size, 
        num_heads=args.num_heads, 
        label_size=args.cond_size,
    )
    args.logger.info(f"Dataset contains {len(train_set):,} images ({args.data_path})")
    return train_set, model, args


def prepare_dataloader(args, dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset,
                                   shuffle=True,
                                   seed=args.global_seed),
        num_workers=args.num_workers,
        drop_last=True,
    )

def main(world_size: int, 
         available_gpus: list,
         input_args):
    
    # Set up DDP
    dist.init_process_group(backend="nccl", world_size=world_size)
    rank = dist.get_rank()
    device = available_gpus[rank]
    seed = input_args.global_seed * dist.get_world_size() + rank
    print("Rank: ", rank, " | Device: ", device, " | Seed: ", seed)
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # set up output folder and logger
    if rank == 0:
        print("Rank 0 mkdir & set up logger...")
        # mkdir for logs and checkpoints
        os.makedirs(input_args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{input_args.results_dir}/*"))
        input_args.experiment_dir = f"{input_args.results_dir}/{experiment_index:03d}"  # Create an experiment folder
        input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(input_args.checkpoint_dir, exist_ok=True)
        os.makedirs(f"{input_args.experiment_dir}/samples", exist_ok=True)      # Store sampling results
        input_args.logger = create_logger(input_args.experiment_dir)
        input_args.logger.info(f"Experiment directory created at {input_args.experiment_dir}")
    else:
        input_args.logger=create_logger(None)
    input_args.logger.info(f"Rank: {rank} | Device: {device} | Seed: {seed}")
    
    # set up training objects
    dataset, model, args = load_train_objs(input_args)
    input_args.logger.info(f"Dataset, model, and args finished loading.")
    train_data = prepare_dataloader(args, dataset, 
                                    int(args.global_batch_size // dist.get_world_size()))
    input_args.logger.info(f"Dataloader finished loading.")
    trainer = Trainer(model, train_data, 
                      rank, int(device.split(":")[-1]), 
                      args)
    input_args.logger.info(f"Trainer finished loading.")
    input_args.logger.info(f"Starting...")
    trainer.train(args.total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # data related arguments
    parser.add_argument("--expr_name", type=str, default="kidney")
    parser.add_argument("--data_path", type=str, default="./hest1k_datasets/kidney/", help="Dataset path")
    parser.add_argument("--results_dir", type=str, default="./kidney_results/runs_loss/", help="Path to hold runs")
    parser.add_argument("--slide_out", type=str, default="NCBI697", help="Test slide ID. Multiple slides separated by comma.") 
    parser.add_argument("--folder_list_filename", type=str, default="all_slide_lst.txt", help="A txt file listing file names for all training and testing slides in the dataset")
    parser.add_argument("--gene_list_filename", type=str, default="HMHVG.txt", help="Selected gene list")
    parser.add_argument("--num_aug_ratio", type=int, default=4, help="Image augmentation ratio (int)")
    
    # model related arguments
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--DiT_num_blocks", type=int, default=, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=, help="DiT heads")
    # training related arguments
    parser.add_argument("--lr", type=float, default=)
    parser.add_argument("--total_epochs", type=int, default=)
    parser.add_argument("--global_batch_size", type=int, default=)
    parser.add_argument("--global_seed", type=int, default=)
    parser.add_argument("--num_workers", type=int, default=, help="Number of GPUs to run the job")
    parser.add_argument("--ckpt_every", type=int, default=, help="Number of iterations to save checkpoints.")
    parser.add_argument("--path_type", type=str, default="ot", choices=["ot", "i","gauss"], help="Flow path type: 'ot' for OT-CFM (continuous t), 'i' for I-CFM (discrete/fixed t).",
)
    input_args = parser.parse_args()

    ## set up available gpus
    world_size = input_args.num_workers
    ## specify GPU id
    available_gpus = ["cuda:1","cuda:5"] 
    ## or use all available GPU
    # available_gpus = ["cuda:"+str(i) for i in range(world_size)]
    print("Available GPUs: ", available_gpus)
    main(world_size, available_gpus, input_args)
