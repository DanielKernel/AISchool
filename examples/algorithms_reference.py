"""
常见 AI/ML 算法的最小可运行 Python 核心实现，用于考试复习与口述。
运行: python examples/algorithms_reference.py
依赖: numpy
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# K-Means：簇分配 + 簇心更新迭代
# ---------------------------------------------------------------------------
def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (n_samples, n_features)
    返回: centers (k, n_features), labels (n_samples,)
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    centers = X[idx].copy()

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = X[mask].mean(axis=0)

        if np.linalg.norm(new_centers - centers) < tol:
            centers = new_centers
            break
        centers = new_centers

    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return centers, labels


# ---------------------------------------------------------------------------
# KNN：欧氏距离 + 多数投票
# ---------------------------------------------------------------------------
def knn_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x: np.ndarray,
    k: int,
) -> int:
    dists = np.linalg.norm(X_train - x, axis=1)
    idx = np.argsort(dists)[:k]
    labels, counts = np.unique(y_train[idx], return_counts=True)
    return int(labels[np.argmax(counts)])


# ---------------------------------------------------------------------------
# PageRank：幂迭代（列随机矩阵 M，阻尼 alpha）
# ---------------------------------------------------------------------------
def pagerank_power_iteration(
    adj: Dict[int, List[int]],
    n_nodes: int,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    adj[i] 为从 i 出发的有向边目标列表；若无出边则为悬挂节点。
    """
    out_deg = np.array([max(len(adj.get(i, [])), 1) for i in range(n_nodes)], dtype=float)
    M = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        targets = adj.get(i, [])
        if not targets:
            M[:, i] = 1.0 / n_nodes
        else:
            for j in targets:
                M[j, i] += 1.0 / len(targets)

    r = np.ones(n_nodes) / n_nodes
    teleport = (1.0 - alpha) / n_nodes
    for _ in range(max_iter):
        r_new = teleport + alpha * M @ r
        if np.linalg.norm(r_new - r, 1) < tol:
            return r_new
        r = r_new
    return r


# ---------------------------------------------------------------------------
# 决策树：熵、信息增益、单阈值划分（教学用最小递归）
# ---------------------------------------------------------------------------
def _entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-15)))


def information_gain(y: np.ndarray, left_mask: np.ndarray) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    parent_h = _entropy(y)
    n_l, n_r = left_mask.sum(), (~left_mask).sum()
    if n_l == 0 or n_r == 0:
        return 0.0
    h_l = _entropy(y[left_mask])
    h_r = _entropy(y[~left_mask])
    return parent_h - (n_l / n) * h_l - (n_r / n) * h_r


class SimpleDecisionStump:
    """单层：选一个特征阈值使信息增益最大；左右叶为各自多数类。"""

    def __init__(self) -> None:
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.label_left: int = 0
        self.label_right: int = 0

    def _majority(self, labels: np.ndarray) -> int:
        vals, counts = np.unique(labels, return_counts=True)
        return int(vals[np.argmax(counts)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        best_gain = -1.0
        n_features = X.shape[1]
        for j in range(n_features):
            values = np.unique(X[:, j])
            for t in values:
                left = X[:, j] <= t
                g = information_gain(y, left)
                if g > best_gain:
                    best_gain = g
                    self.feature_idx = j
                    self.threshold = float(t)
                    self.label_left = self._majority(y[left])
                    self.label_right = self._majority(y[~left])

    def predict_row(self, x: np.ndarray) -> int:
        return self.label_left if x[self.feature_idx] <= self.threshold else self.label_right


# ---------------------------------------------------------------------------
# 2D 单通道卷积（valid）与 Max Pooling
# ---------------------------------------------------------------------------
def conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    image: (H, W), kernel: (kh, kw), valid 模式无 padding。
    """
    H, W = image.shape
    kh, kw = kernel.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i : i + kh, j : j + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def max_pool2d(x: np.ndarray, pool_h: int, pool_w: int, stride: int) -> np.ndarray:
    H, W = x.shape
    out_h = (H - pool_h) // stride + 1
    out_w = (W - pool_w) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = x[
                i * stride : i * stride + pool_h,
                j * stride : j * stride + pool_w,
            ]
            out[i, j] = patch.max()
    return out


# ---------------------------------------------------------------------------
# 二分图：BFS 二染色
# ---------------------------------------------------------------------------
def is_bipartite(n: int, edges: List[Tuple[int, int]]) -> bool:
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    color: List[Optional[int]] = [None] * n
    for start in range(n):
        if color[start] is not None:
            continue
        color[start] = 0
        q: deque[int] = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if color[v] is None:
                    color[v] = 1 - int(color[u])  # type: ignore
                    q.append(v)
                elif color[v] == color[u]:
                    return False
    return True


# ---------------------------------------------------------------------------
# Kruskal：并查集 + 按权边排序
# ---------------------------------------------------------------------------
class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
        return True


def kruskal_mst(n: int, edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """
    edges: (u, v, weight)，无向图；返回 MST 边列表。
    """
    edges_sorted = sorted(edges, key=lambda e: e[2])
    dsu = _DSU(n)
    mst: List[Tuple[int, int, float]] = []
    for u, v, w in edges_sorted:
        if dsu.union(u, v):
            mst.append((u, v, w))
        if len(mst) == n - 1:
            break
    return mst


# ---------------------------------------------------------------------------
# 线性回归：批量梯度下降
# ---------------------------------------------------------------------------
def linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 500,
) -> np.ndarray:
    """
    X: (m, d)，可自己加一列全 1 作为偏置；y: (m,)
    返回 w: (d,)
    """
    m, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        pred = X @ w
        grad = (2.0 / m) * (X.T @ (pred - y))
        w -= lr * grad
    return w


# ---------------------------------------------------------------------------
# SMOTE：少数类过采样（简化版，连续特征）
# ---------------------------------------------------------------------------
def smote(
    X_minority: np.ndarray,
    k: int = 3,
    n_synthetic: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """
    X_minority: (n_minor, n_features)，仅少数类样本。
    返回合成样本 (n_synthetic, n_features)。
    """
    rng = np.random.default_rng(seed)
    n, d = X_minority.shape
    if n < 2:
        raise ValueError("SMOTE 需要至少 2 个少数类样本")
    k_eff = min(k, n - 1)
    synth = np.zeros((n_synthetic, d))
    for t in range(n_synthetic):
        i = rng.integers(0, n)
        dists = np.linalg.norm(X_minority - X_minority[i], axis=1)
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        j = int(rng.choice(nn_idx))
        lam = rng.random()
        synth[t] = X_minority[i] + lam * (X_minority[j] - X_minority[i])
    return synth


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> None:
    # K-Means
    rng = np.random.default_rng(42)
    c1 = rng.normal(0, 0.3, (30, 2))
    c2 = rng.normal(3, 0.3, (30, 2))
    Xk = np.vstack([c1, c2])
    centers, labels = kmeans(Xk, k=2, seed=1)
    assert centers.shape == (2, 2) and labels.shape == (60,)

    # KNN
    yk = np.array([0] * 30 + [1] * 30)
    pred = knn_classify(Xk, yk, np.array([0.0, 0.0]), k=5)
    assert pred == 0

    # PageRank：简单链
    pr = pagerank_power_iteration({0: [1], 1: [2], 2: [0]}, 3)
    assert abs(pr.sum() - 1.0) < 1e-6

    # 信息增益 + 决策桩
    Xd = np.array([[1], [2], [3], [10]])
    yd = np.array([0, 0, 1, 1])
    stump = SimpleDecisionStump()
    stump.fit(Xd, yd)
    assert stump.predict_row(np.array([1.5])) in (0, 1)

    # 卷积 / 池化
    img = np.arange(16, dtype=float).reshape(4, 4)
    ker = np.ones((2, 2))
    c = conv2d_valid(img, ker)
    assert c.shape == (3, 3)
    p = max_pool2d(img, 2, 2, stride=2)
    assert p.shape == (2, 2)

    # 二分图
    assert is_bipartite(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) is True
    assert is_bipartite(3, [(0, 1), (1, 2), (2, 0)]) is False

    # Kruskal
    mst = kruskal_mst(4, [(0, 1, 1), (1, 2, 2), (0, 2, 3), (2, 3, 1)])
    assert len(mst) == 3

    # 梯度下降（含偏置列）
    Xm = np.column_stack([np.linspace(0, 1, 20), np.ones(20)])
    ym = 3 * Xm[:, 0] + 2 + rng.normal(0, 0.05, 20)
    w = linear_regression_gd(Xm, ym, lr=0.5, epochs=2000)
    assert abs(w[0] - 3.0) < 0.2

    # SMOTE
    Xmin = rng.normal(size=(8, 3))
    syn = smote(Xmin, k=3, n_synthetic=5, seed=0)
    assert syn.shape == (5, 3)

    print("algorithms_reference: 所有自检通过。")


if __name__ == "__main__":
    _self_check()
