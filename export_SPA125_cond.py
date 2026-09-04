# 构造测试条件向量
import torch
import numpy as np
import anndata
import pandas as pd

data_path = "./hest1k_datasets/kidney/"
resolutions = [224, 112, 56]

def get_suffix(res):
    return "" if res == 224 else f"_{res}"

slide_out = "NCBI692"

# 1. 读取测试 slide 的 h5ad，主要是为了拿到 spot 数和顺序
test_adata = anndata.read_h5ad(data_path + "st/" + slide_out + ".h5ad")
num_spots = test_adata.shape[0]

# 2. 按训练时的方式加载多分辨率 CONCH+UNI embedding
img_ebd_list = []
for res in resolutions:
    suffix = get_suffix(res)
    img_ebd_uni = torch.load(
        data_path + f"processed_data/1spot_uni_ebd/{slide_out}_uni{suffix}.pt",
        map_location="cpu",
    )
    img_ebd_conch = torch.load(
        data_path + f"processed_data/1spot_conch_ebd/{slide_out}_conch{suffix}.pt",
        map_location="cpu",
    )
    img_ebd_list.extend([img_ebd_uni, img_ebd_conch])

slide_img_ebd = torch.cat(img_ebd_list, axis=1)  # (num_spots, 4608)

assert slide_img_ebd.shape[0] == num_spots
torch.save(slide_img_ebd, "./kidney_results/.../NCBI692_cond.pt")
print("Saved cond to ./kidney_results/.../NCBI692_cond.pt",
      "shape =", slide_img_ebd.shape)
