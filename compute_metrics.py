#!/usr/bin/env python3
"""
CUDA_VISIBLE_DEVICES=5,7 python compute_metrics.py --sample_path ./ablation_results/kidney/full_seed42/samples/generated_full_0375000_20sample.pt --data_path ./hest1k_datasets/kidney/processed_data/ --ori_st_path ./hest1k_datasets/kidney/st/ --slide_out NCBI697 --gene_list HMHVG.txt --topk 10 50 200
compute_metrics.py

Compute spatial gene-expression prediction metrics (PCC-k, MSE, MAE, RVD)
for GeneFlow / Stem outputs. Ported from eval.ipynb.

Pipeline:
  1. Load the generated samples (.pt) produced by stem_sample_flow.py
     (or stem_sample.py).
  2. Average the repeated samples per spot -> one prediction per spot.
  3. Load the held-out test slide ground truth (h5ad), keep the selected
     genes, and log2(count + 1) transform it.
  4. Compute per-gene Pearson correlation + MSE / MAE / RVD.

Assumptions (same as the original notebook):
  - Predictions are already in log2 space (that is what the model generates).
  - Predicted rows are in the SAME spot order as the test h5ad, with each
    spot's repeated samples stored in contiguous blocks
    (rows [0:num_rep] -> spot 0, [num_rep:2*num_rep] -> spot 1, ...).

Example:
  python compute_metrics.py                      # uses the defaults below
  python compute_metrics.py --sample_path <...>  # point to another output
  python compute_metrics.py --plot               # also save the variation figure
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import anndata


def densify(x):
    """Return a dense numpy array whether x is a scipy sparse matrix or dense."""
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def load_ground_truth(ori_st_path, slide_out, selected_genes):
    adata = anndata.read_h5ad(os.path.join(ori_st_path, slide_out + ".h5ad"))
    var_set = set(map(str, adata.var_names))
    missing = [g for g in selected_genes if g not in var_set]
    if missing:
        raise ValueError(
            f"{len(missing)} selected genes are missing from {slide_out}.h5ad "
            f"(e.g. {missing[:5]}). Make sure --gene_list matches the list used "
            f"for training / sampling."
        )
    gt_df = pd.DataFrame(
        densify(adata.X), columns=adata.var_names, index=adata.obs_names
    ).loc[:, selected_genes]
    gt_log = np.log2(gt_df + 1).copy()           # (n_test, n_genes), log2 space
    return adata, gt_log


def average_samples(pred, n_test):
    """Average per-spot repeated samples (contiguous blocks) into (n_test, G)."""
    if not torch.is_tensor(pred):
        pred = torch.as_tensor(pred)

    # accept (N, G) or (N, 1, G)
    if pred.dim() == 3 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if pred.dim() != 2:
        raise ValueError(
            f"Unexpected prediction shape {tuple(pred.shape)}; expected (N, G) or (N, 1, G)."
        )

    n_pred = pred.shape[0]
    if n_pred < n_test:
        raise ValueError(f"Fewer predicted rows ({n_pred}) than test spots ({n_test}).")
    num_rep = n_pred // n_test
    if n_pred % n_test != 0:
        print(
            f"[warn] N_pred ({n_pred}) is not a clean multiple of N_test ({n_test}); "
            f"using num_rep={num_rep} and ignoring the trailing "
            f"{n_pred - num_rep * n_test} rows."
        )

    pred = pred.float()
    pred_avg = torch.zeros((n_test, pred.shape[1]), dtype=pred.dtype)
    for i in range(n_test):
        start = i * num_rep
        end = min((i + 1) * num_rep, n_pred)
        pred_avg[i] = pred[start:end].mean(dim=0)
    return pred_avg.cpu().numpy(), num_rep


def compute_metrics(gt_log, pred_avg, topk):
    gt = gt_log.values                           # (n_test, n_genes), log2 space
    n_genes = gt.shape[1]

    # per-gene Pearson correlation between GT and averaged prediction
    corr = np.array(
        [np.corrcoef(gt[:, g], pred_avg[:, g])[0, 1] for g in range(n_genes)],
        dtype=float,
    )
    valid = corr[~np.isnan(corr)]
    if len(valid) < n_genes:
        print(
            f"[warn] {n_genes - len(valid)} gene(s) had undefined correlation "
            f"(zero variance) and were skipped in PCC."
        )

    corr_desc = np.sort(valid)[::-1]
    metrics = {}
    for k in topk:
        kk = min(k, len(corr_desc))
        metrics[f"PCC-{k}"] = float(np.mean(corr_desc[:kk]))

    metrics["MSE"] = float(np.mean((gt - pred_avg) ** 2))
    metrics["MAE"] = float(np.mean(np.abs(gt - pred_avg)))

    pred_var = np.var(pred_avg, axis=0)
    gt_var = np.var(gt, axis=0)
    nz = gt_var > 0                              # avoid divide-by-zero
    metrics["RVD"] = float(np.mean(((pred_var[nz] - gt_var[nz]) ** 2) / (gt_var[nz] ** 2)))

    return metrics


def save_variation_plot(gt_log, pred_avg, out_png):
    import matplotlib
    matplotlib.use("Agg")                        # headless: save instead of show
    import matplotlib.pyplot as plt

    gt = gt_log.values
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    pm, gm = pred_avg.mean(0), gt.mean(0)
    order = np.argsort(gm)
    axs[0, 0].plot(np.arange(len(gm)), (gm / gm.sum())[order], c="b", label="Ground Truth")
    axs[0, 0].scatter(np.arange(len(pm)), (pm / pm.sum())[order], s=5, c="orange", label="Predicted")
    axs[0, 0].set(title="Normalized Mean", xlabel="gene index ordered by mean", ylabel="normalized mean")
    axs[0, 0].legend()

    axs[1, 0].plot(np.arange(len(gm)), gm[order], c="b")
    axs[1, 0].scatter(np.arange(len(pm)), pm[order], s=5, c="orange")
    axs[1, 0].set(title="Absolute Mean", xlabel="gene index ordered by mean", ylabel="absolute mean")

    pv, gv = pred_avg.var(0), gt.var(0)
    order = np.argsort(gv)
    axs[0, 1].plot(np.arange(len(gv)), (gv / gv.sum())[order], c="b")
    axs[0, 1].scatter(np.arange(len(pv)), (pv / pv.sum())[order], s=5, c="orange")
    axs[0, 1].set(title="Normalized Variance", xlabel="gene index ordered by var", ylabel="normalized variance")

    axs[1, 1].plot(np.arange(len(gv)), gv[order], c="b")
    axs[1, 1].scatter(np.arange(len(pv)), pv[order], s=5, c="orange")
    axs[1, 1].set(title="Absolute Variance", xlabel="gene index ordered by var", ylabel="absolute variance")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Gene-variation figure saved to: {out_png}")


def main():
    p = argparse.ArgumentParser(
        description="Compute ST prediction metrics (PCC-k, MSE, MAE, RVD)."
    )
    p.add_argument("--sample_path", type=str,
                   default="./kidney_results/runs_0/000/samples/generated_samples_0400000_20sample.pt",
                   help="Path to generated samples .pt")
    p.add_argument("--data_path", type=str, default="./hest1k_datasets/kidney/processed_data/",
                   help="Processed-data dir (holds all_slide_lst.txt and the gene list)")
    p.add_argument("--ori_st_path", type=str, default="./hest1k_datasets/kidney/st/",
                   help="Dir holding the original ST h5ad files")
    p.add_argument("--slide_out", type=str, default="NCBI697", help="Held-out test slide ID")
    p.add_argument("--gene_list", type=str, default="HMHVG.txt",
                   help="Gene list filename (under data_path)")
    p.add_argument("--topk", type=int, nargs="+", default=[10, 50, 200],
                   help="Top-k values for PCC (default: 10 50 200)")
    p.add_argument("--plot", action="store_true",
                   help="Also save the gene mean/variance figure as a PNG")
    p.add_argument("--no_save", action="store_true",
                   help="Do not write a metrics JSON file")
    args = p.parse_args()

    # selected genes
    selected_genes = list(np.genfromtxt(os.path.join(args.data_path, args.gene_list), dtype=str))
    print(f"Selected genes: {len(selected_genes)} (from {args.gene_list})")

    # sanity: confirm the slide is actually held out
    slide_lst_file = os.path.join(args.data_path, "all_slide_lst.txt")
    if os.path.isfile(slide_lst_file):
        slides = list(np.genfromtxt(slide_lst_file, dtype=str))
        print(f"{args.slide_out} present in slide list: {args.slide_out in slides}")

    # ground truth
    _, gt_log = load_ground_truth(args.ori_st_path, args.slide_out, selected_genes)
    print(f"Test count matrix shape: {gt_log.shape}  (spots x genes)")

    # predictions
    pred = torch.load(args.sample_path, map_location="cpu")
    print(f"Generated samples shape: {tuple(pred.shape)}")

    n_test = gt_log.shape[0]
    pred_avg, num_rep = average_samples(pred, n_test)
    print(f"N_test={n_test}, N_pred={int(np.asarray(pred).shape[0]) if not torch.is_tensor(pred) else pred.shape[0]}, "
          f"samples per spot (num_rep)={num_rep}")

    if pred_avg.shape[1] != len(selected_genes):
        raise AssertionError(
            f"Prediction gene dim ({pred_avg.shape[1]}) != number of selected genes "
            f"({len(selected_genes)}). Check that --gene_list matches the list used at sampling time."
        )

    # metrics
    metrics = compute_metrics(gt_log, pred_avg, args.topk)
    print("\n===== Metrics =====")
    for k in args.topk:
        print(f"PCC-{k:<3}: {metrics[f'PCC-{k}']:.4f}")
    print(f"MSE   : {metrics['MSE']:.4f}")
    print(f"MAE   : {metrics['MAE']:.4f}")
    print(f"RVD   : {metrics['RVD']:.4f}")

    # outputs
    out_dir = os.path.dirname(os.path.abspath(args.sample_path))
    stem = os.path.splitext(os.path.basename(args.sample_path))[0]

    if not args.no_save:
        out_json = os.path.join(out_dir, f"metrics_{stem}.json")
        with open(out_json, "w") as f:
            json.dump(
                {"slide_out": args.slide_out, "sample_path": args.sample_path,
                 "num_rep": num_rep, **metrics},
                f, indent=2,
            )
        print(f"\nMetrics saved to: {out_json}")

    if args.plot:
        save_variation_plot(gt_log, pred_avg, os.path.join(out_dir, f"gene_variation_{stem}.png"))


if __name__ == "__main__":
    main()
