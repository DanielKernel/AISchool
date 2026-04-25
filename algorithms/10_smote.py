"""
SMOTE 过采样 — 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.smote_n_new
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from aischool.algorithm_core import smote_n_new


def smote(X_minority, n_new_samples, k=5):
    return smote_n_new(X_minority, n_new_samples, k=k)


if __name__ == "__main__":
    np.random.seed(42)

    X_majority = np.random.randn(100, 2) + [5, 5]
    X_minority = np.random.randn(10, 2) + [0, 0]

    print("SMOTE 过采样:")
    print("=" * 40)
    print(f"原始少数类样本数: {len(X_minority)}")
    print(f"原始多数类样本数: {len(X_majority)}")
    print(f"不平衡比例: 1:{len(X_majority)//len(X_minority)}")

    new_samples = smote(X_minority, n_new_samples=90, k=5)
    X_minority_balanced = np.vstack([X_minority, new_samples])

    print(f"\nSMOTE 后少数类样本数: {len(X_minority_balanced)}")
    print(f"平衡后比例: 1:{len(X_majority)//len(X_minority_balanced)}")

    print(f"\n新生成样本示例 (前3个):")
    for s in new_samples[:3]:
        print(f"  ({s[0]:.2f}, {s[1]:.2f})")

    print()
    print("--- 理解辅助：一次插值过程追踪 ---")
    np.random.seed(0)
    idx = 0
    sample = X_minority[idx]
    distances = np.linalg.norm(X_minority - sample, axis=1)
    nn_indices = np.argsort(distances)[1:6]
    nn_idx = nn_indices[0]
    neighbor = X_minority[nn_idx]
    lam = 0.4
    x_new = sample + lam * (neighbor - sample)
    print(f"  选中样本 x = ({sample[0]:.2f}, {sample[1]:.2f})")
    print(f"  最近邻 x_nn = ({neighbor[0]:.2f}, {neighbor[1]:.2f})")
    print(f"  λ = {lam}")
    print(f"  x_new = x + {lam} * (x_nn - x) = ({x_new[0]:.2f}, {x_new[1]:.2f})")
    print(f"  → 新样本落在 x 和 x_nn 的连线上，距 x 约 {lam*100:.0f}% 处")
