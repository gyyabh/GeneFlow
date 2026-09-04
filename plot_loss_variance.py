# plot_loss_variance.py
# -----------------------------------------------------------------------------
# 画「DT-CFM(离散时间)vs Continuous-CFM(连续时间)」的训练 loss 对比曲线,
# 并量化各自的 loss 抖动幅度(滚动标准差),支撑方案中
# 「离散时间采样降低优化方差」这一论断(论文 Section 1 的核心卖点之一)。
#
# 输入:train_ablation.py 落盘的 loss_history.csv
#   <results_dir>/full_seed42/loss_history.csv         (DT-CFM)
#   <results_dir>/continuous_cfm_seed42/loss_history.csv (Continuous-CFM)
#
# 用法:
#   python plot_loss_variance.py \
#       --results_dir ./ablation_results/kidney/ --seed 42 \
#       --out_png ./ablation_results/kidney/loss_variance.png
# -----------------------------------------------------------------------------
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_loss(results_dir, variant, seed):
    path = os.path.join(results_dir, f"{variant}_seed{seed}", "loss_history.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 {path};确认该 variant 已训练并落盘 loss_history.csv")
    return pd.read_csv(path)


def rolling_std(y, win=10):
    s = pd.Series(y).rolling(win, min_periods=1).std().fillna(0).values
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--win", type=int, default=10, help="滚动标准差窗口")
    p.add_argument("--out_png", type=str, default="./loss_variance.png")
    args = p.parse_args()

    dt = load_loss(args.results_dir, "full", args.seed)            # DT-CFM
    ct = load_loss(args.results_dir, "continuous_cfm", args.seed)  # Continuous-CFM

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左:loss 曲线
    axs[0].plot(dt["step"], dt["avg_loss"], label="DT-CFM (discrete t)", c="tab:blue")
    axs[0].plot(ct["step"], ct["avg_loss"], label="Continuous-CFM (t~U[0,1])", c="tab:orange", alpha=0.8)
    axs[0].set(title="Training Loss", xlabel="step", ylabel="loss")
    axs[0].legend()

    # 右:滚动标准差(方差代理)
    axs[1].plot(dt["step"], rolling_std(dt["avg_loss"].values, args.win),
                label="DT-CFM", c="tab:blue")
    axs[1].plot(ct["step"], rolling_std(ct["avg_loss"].values, args.win),
                label="Continuous-CFM", c="tab:orange", alpha=0.8)
    axs[1].set(title=f"Loss rolling std (win={args.win})", xlabel="step", ylabel="rolling std")
    axs[1].legend()

    # 文本汇总:整体 loss 标准差
    dt_std = float(np.std(dt["avg_loss"].values))
    ct_std = float(np.std(ct["avg_loss"].values))
    fig.suptitle(f"Overall loss std — DT-CFM: {dt_std:.4f}  |  Continuous-CFM: {ct_std:.4f}")

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved: {args.out_png}")
    print(f"DT-CFM loss std={dt_std:.4f}  Continuous-CFM loss std={ct_std:.4f}")


if __name__ == "__main__":
    main()
