"""
K-Means 聚类算法
================

【算法背景】
K-Means 是最经典的无监督学习聚类算法，由 Stuart Lloyd 于 1957 年提出。
目标：将 n 个数据点划分为 K 个簇，使得每个点属于离它最近的簇心所代表的簇，
      从而最小化簇内平方和 (Within-Cluster Sum of Squares, WCSS)。

【适用场景】
- 客户分群、图像颜色压缩、文档聚类、异常检测的预处理

【算法流程（4 步循环）】
    ┌──────────────────────────────┐
    │ Step 1: 随机选 K 个样本作为初始簇心 │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: 对每个样本，计算到 K 个   │
    │   簇心的距离，分配到最近的簇    │◄──┐
    └──────────┬───────────────────┘   │
               ▼                       │
    ┌──────────────────────────────┐   │
    │ Step 3: 对每个簇，重新计算簇心    │   │
    │   (簇内所有点的均值)            │   │
    └──────────┬───────────────────┘   │
               ▼                       │
    ┌──────────────────────────────┐   │
    │ Step 4: 簇心变了吗？            │   │
    │   是 → 回到 Step 2              │───┘
    │   否 → 收敛，结束               │
    └──────────────────────────────┘

【核心公式】
  距离计算:  d(x, c) = ||x - c||₂ = sqrt(Σ(xᵢ - cᵢ)²)
  簇心更新:  c_new = (1/|S|) * Σ x    (S 是属于该簇的所有点)

【口诀】初始化 → 分配 → 更新 → 重复

【记忆要点 - 考试必背】
  1. 初始化: np.random.choice 随机选 K 个不重复索引
  2. 距离计算: np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
     - X[:, np.newaxis] 把 X 从 (n, d) 扩展为 (n, 1, d)
     - 减去 centroids (k, d) 后广播为 (n, k, d)
     - axis=2 对特征维度求范数 → 得到 (n, k) 的距离矩阵
  3. 分配: np.argmin(distances, axis=1) → 每行取最小距离的索引
  4. 更新: X[labels == i].mean(axis=0) → 布尔索引选出簇内点再求均值
  5. 收敛判断: np.allclose(old, new) → 簇心几乎不动就停止

【易错点】
  - 空簇处理：某个簇可能没有点被分配到，此时保持原簇心不变
  - K 值选择：实际应用中常用「肘部法则」(Elbow Method)，但考试一般给定 K
  - 初始化敏感：结果受初始簇心影响，实际中常用 K-Means++ 改进

【复杂度】
  时间: O(n * k * d * T)  n=样本数, k=簇数, d=维度, T=迭代次数
  空间: O(n * k)          距离矩阵
"""
import numpy as np


def kmeans(X, k, max_iters=100):
    """
    参数:
        X: 数据矩阵, shape (n_samples, n_features)
        k: 簇的数量
        max_iters: 最大迭代次数
    返回:
        centroids: 最终簇心, shape (k, n_features)
        labels: 每个样本的簇标签, shape (n_samples,)
    """
    n = X.shape[0]

    # Step 1: 随机选 k 个样本作为初始簇心
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices].copy()

    for _ in range(max_iters):
        # Step 2: 计算每个点到每个簇心的距离，分配到最近簇心
        # distances shape: (n_samples, k)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 3: 更新簇心为簇内所有点的均值
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        # Step 4: 检查是否收敛（簇心不再变化）
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# ========== 运行示例 ==========
if __name__ == "__main__":
    np.random.seed(42)
    # 生成 3 簇数据
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

    # 手动追踪一次迭代，帮助理解
    print("--- 理解辅助：距离计算过程 ---")
    sample = data[0]
    print(f"样本 data[0] = ({sample[0]:.2f}, {sample[1]:.2f})")
    for i, c in enumerate(centroids):
        d = np.linalg.norm(sample - c)
        print(f"  到簇心 {i} 的距离: {d:.2f}")
    print(f"  → 分配到簇 {labels[0]}")
