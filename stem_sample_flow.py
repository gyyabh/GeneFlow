# 流匹配推理脚本
# python stem_sample_flow.py --checkpoint ./kidney_results/runs_0/004/checkpoints/0400000.pt --cond_path ./kidney_results/runs_ot/002/samples/NCBI697_cond.pt --output_dir ./kidney_results/runs_0/000/samples/ --input_gene_size 200 --cond_size 4608 --DiT_num_blocks 12 --hidden_size 384 --num_heads 6 --num_steps 100 --device cuda:0
import os
import torch
import argparse
import numpy as np

from Stem.models import Stem_models
from Stem.train_helper import flow_sample


def load_checkpoint(checkpoint_path, model):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./kidney_results/runs_loss/000/checkpoints/0025000.pt", help="Path to EMA checkpoint .pt")
    parser.add_argument("--cond_path", type=str, default="./kidney_results/runs_ot/002/samples/NCBI697_cond.pt", help="Path to condition embedding .pt")
    parser.add_argument("--output_dir", type=str, default="./kidney_results/runs_loss/000/samples/", help="Directory to save results")
    parser.add_argument("--sample_num_per_cond", type=int, default=20, help="Used for filename only")
    parser.add_argument("--model", type=str, default="Stem")
    parser.add_argument("--input_gene_size", type=int, default=200)
    parser.add_argument("--cond_size", type=int, default=4608)
    parser.add_argument("--DiT_num_blocks", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:4")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_name = os.path.basename(args.checkpoint).split('.')[0]
    base_name = f"generated_samples_{ckpt_name}_{args.sample_num_per_cond}sample"
    
    model = Stem_models[args.model](
        input_size=args.input_gene_size,
        depth=args.DiT_num_blocks,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        label_size=args.cond_size,
    ).to(device)

    model = load_checkpoint(args.checkpoint, model)

    cond_tensor = torch.load(args.cond_path, map_location="cpu")
    cond = cond_tensor.float().to(device)

    # sample
    x0 = flow_sample(model, cond, num_steps=args.num_steps, device=device, show_progress=True)
    x0_cpu = x0.cpu()

    # 5. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存为 .pt
    pt_path = os.path.join(args.output_dir, f"{base_name}.pt")
    torch.save(x0_cpu, pt_path)
    
    # 保存为 .npy (可选)
    npy_path = os.path.join(args.output_dir, f"{base_name}.npy")
    np.save(npy_path, x0_cpu.numpy())

    print(f"Results saved to:\nPT: {pt_path}\nNPY: {npy_path}\nShape: {x0_cpu.shape}")


if __name__ == "__main__":
    main()