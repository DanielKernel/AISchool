"""
K-Means 聚类算法
================
核心思想：随机初始化 K 个簇心 → 将每个点分配到最近簇心 → 更新簇心为簇内均值 → 重复直到收敛

口诀：初始化 → 分配 → 更新 → 重复
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
    print("簇心坐标:")
    for i, c in enumerate(centroids):
        print(f"  簇 {i}: ({c[0]:.2f}, {c[1]:.2f})")
    print(f"标签分布: {np.bincount(labels)}")
