"""
K-Means 聚类算法 — 详解与口诀见根目录 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.kmeans
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from aischool.algorithm_core import kmeans as _kmeans_core


def kmeans(X, k, max_iters=100):
    """参数与行为与速通版一致；随机种子不固定（与 numpy 全局 RNG 一致）。"""
    return _kmeans_core(X, K=k, max_iter=max_iters, random_state=None)


if __name__ == "__main__":
    np.random.seed(42)
    data = np.vstack([
        np.random.randn(30, 2) + [0, 0],
        np.random.randn(30, 2) + [5, 5],
        np.random.randn(30, 2) + [10, 0],
    ])

    centroids, labels = kmeans(data, k=3)
    print("K-Means 聚类结果:")
    print("=" * 40)
    print("簇心坐标:")
    for i, c in enumerate(centroids):
        print(f"  簇 {i}: ({c[0]:.2f}, {c[1]:.2f})")
    print(f"标签分布: {np.bincount(labels)}")
    print()
    print("--- 理解辅助：距离计算过程 ---")
    sample = data[0]
    print(f"样本 data[0] = ({sample[0]:.2f}, {sample[1]:.2f})")
    for i, c in enumerate(centroids):
        d = np.linalg.norm(sample - c)
        print(f"  到簇心 {i} 的距离: {d:.2f}")
    print(f"  → 分配到簇 {labels[0]}")
