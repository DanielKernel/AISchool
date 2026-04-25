"""
10 个核心算法的唯一实现源（供根目录演示、examples 自检、algorithms/*.py 引用）。
依赖：numpy；图算法中二分图与 PageRank（字典版）仅用标准库。
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 距离
# ---------------------------------------------------------------------------


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离（L2）。"""
    return float(np.sqrt(np.sum((a - b) ** 2)))


# ---------------------------------------------------------------------------
# 1. K-Means
# ---------------------------------------------------------------------------


def kmeans_vectorized(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """向量化 K-Means（examples 自检用）。"""
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


def kmeans(
    X: np.ndarray,
    K: int,
    max_iter: int = 100,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    idx = np.random.choice(n_samples, K, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        new_labels = np.array(
            [np.argmin([euclidean_distance(x, c) for c in centers]) for x in X]
        )
        new_centers = np.array(
            [
                X[new_labels == k].mean(axis=0)
                if np.any(new_labels == k)
                else centers[k]
                for k in range(K)
            ]
        )
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
        labels = new_labels

    return centers, new_labels


# ---------------------------------------------------------------------------
# 2. KNN（批量预测）
# ---------------------------------------------------------------------------


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    K: int = 3,
) -> np.ndarray:
    predictions = []
    for x in X_test:
        distances = [euclidean_distance(x, xi) for xi in X_train]
        k_indices = np.argsort(distances)[:K]
        k_labels = y_train[k_indices]
        values, counts = np.unique(k_labels, return_counts=True)
        predictions.append(values[np.argmax(counts)])
    return np.array(predictions)


def knn_predict_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 3,
) -> int:
    """单点 KNN（与 algorithms/02 接口一致）。"""
    distances = np.linalg.norm(X_train - x_query, axis=1)
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_nearest_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return int(most_common[0][0])


def knn_predict_batch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    return np.array([knn_predict_one(X_train, y_train, x, k) for x in X_test])


# ---------------------------------------------------------------------------
# 3. PageRank（字典邻接表，与 graph_algorithms 一致）
# ---------------------------------------------------------------------------


def pagerank(
    graph: Dict[Any, List[Any]],
    d: float = 0.85,
    max_iter: int = 100,
    epsilon: float = 1e-6,
) -> Dict[Any, float]:
    nodes = list(graph.keys())
    N = len(nodes)
    rank = {node: 1.0 / N for node in nodes}
    in_links = {node: [] for node in nodes}
    out_degree = {node: len(graph[node]) for node in nodes}
    for node in nodes:
        for neighbor in graph[node]:
            in_links[neighbor].append(node)

    for iteration in range(max_iter):
        new_rank = {}
        for node in nodes:
            teleport = (1 - d) / N
            link_sum = sum(
                rank[src] / out_degree[src]
                for src in in_links[node]
                if out_degree[src] > 0
            )
            new_rank[node] = teleport + d * link_sum
        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < epsilon:
            break
    return rank


def pagerank_adjacency(
    adj_matrix: np.ndarray,
    damping: float = 0.85,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """邻接矩阵版 PageRank：adj[i,j]=1 表示 i→j。"""
    n = adj_matrix.shape[0]
    out_degree = adj_matrix.sum(axis=1, keepdims=True)
    out_degree[out_degree == 0] = 1
    M = (adj_matrix / out_degree).T
    ranks = np.ones(n) / n
    for _ in range(max_iters):
        new_ranks = (1 - damping) / n + damping * M @ ranks
        if np.linalg.norm(new_ranks - ranks) < tol:
            return new_ranks
        ranks = new_ranks
    return ranks


# ---------------------------------------------------------------------------
# 4. 决策树：熵与信息增益（离散特征列）
# ---------------------------------------------------------------------------


def entropy(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def information_gain(X_col: np.ndarray, y: np.ndarray) -> float:
    parent_entropy = entropy(y)
    n = len(y)
    weighted_child_entropy = 0.0
    for val in np.unique(X_col):
        mask = X_col == val
        child_y = y[mask]
        weight = len(child_y) / n
        weighted_child_entropy += weight * entropy(child_y)
    return parent_entropy - weighted_child_entropy


def best_split_feature(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    n_features = X.shape[1]
    gains = [information_gain(X[:, j], y) for j in range(n_features)]
    best_feat_idx = int(np.argmax(gains))
    return best_feat_idx, gains[best_feat_idx]


# ---------------------------------------------------------------------------
# 5–6. 卷积与池化
# ---------------------------------------------------------------------------


def conv2d(
    input_map: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    if padding > 0:
        input_map = np.pad(input_map, padding, mode="constant", constant_values=0)
    H_pad, W_pad = input_map.shape
    kH, kW = kernel.shape
    out_H = (H_pad - kH) // stride + 1
    out_W = (W_pad - kW) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            region = input_map[
                i * stride : i * stride + kH, j * stride : j * stride + kW
            ]
            output[i, j] = np.sum(region * kernel)
    return output


def max_pooling2d(
    input_map: np.ndarray,
    pool_size: int = 2,
    stride: Optional[int] = None,
) -> np.ndarray:
    if stride is None:
        stride = pool_size
    H, W = input_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            region = input_map[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size,
            ]
            output[i, j] = np.max(region)
    return output


def conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid 卷积（无 padding、步长 1），与自检样例一致。"""
    return conv2d(image, kernel, stride=1, padding=0)


def max_pool2d(
    x: np.ndarray,
    pool_h: int,
    pool_w: int,
    stride: int,
) -> np.ndarray:
    """矩形池化窗口（与 examples 自检一致）。"""
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


def avg_pooling2d(
    input_map: np.ndarray,
    pool_size: int = 2,
    stride: Optional[int] = None,
) -> np.ndarray:
    if stride is None:
        stride = pool_size
    H, W = input_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            region = input_map[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size,
            ]
            output[i, j] = np.mean(region)
    return output


# ---------------------------------------------------------------------------
# 7. 二分图（字典邻接表）
# ---------------------------------------------------------------------------


def is_bipartite(
    graph: Dict[Any, List[Any]],
) -> Tuple[bool, Dict[Any, int]]:
    color: Dict[Any, int] = {}
    for start in graph:
        if start in color:
            continue
        queue = deque([start])
        color[start] = 0
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False, color
    return True, color


# ---------------------------------------------------------------------------
# 8. Kruskal（边格式 (u, v, weight)）
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def kruskal(
    n: int,
    edges: List[Tuple[int, int, float]],
) -> Tuple[List[Tuple[int, int, float]], float]:
    sorted_edges = sorted(edges, key=lambda e: e[2])
    uf = UnionFind(n)
    mst_edges: List[Tuple[int, int, float]] = []
    total_weight = 0.0
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total_weight += w
            if len(mst_edges) == n - 1:
                break
    return mst_edges, total_weight


def kruskal_weight_first(
    n: int,
    edges: List[Tuple[float, int, int]],
) -> Tuple[List[Tuple[int, int, float]], float]:
    """边列表为 (weight, u, v)，与 algorithms/08 示例一致。"""
    as_uvw = [(u, v, w) for w, u, v in edges]
    mst, total = kruskal(n, as_uvw)
    return mst, total


# ---------------------------------------------------------------------------
# 9. 梯度下降（线性回归，显式 w 与 b）
# ---------------------------------------------------------------------------


def gradient_descent_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1000,
) -> Tuple[np.ndarray, float, List[float]]:
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    loss_history: List[float] = []
    for epoch in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y
        loss = float(np.mean(error**2))
        loss_history.append(loss)
        grad_w = (X.T @ error) / n
        grad_b = float(error.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, loss_history


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, List[float]]:
    """X 已含偏置列时的批量梯度下降（与 algorithms/09 一致）。"""
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    losses: List[float] = []
    for _ in range(max_iters):
        y_pred = X @ w
        error = y_pred - y
        loss = float(np.mean(error**2))
        losses.append(loss)
        gradient = (2 / n) * (X.T @ error)
        w = w - lr * gradient
        if len(losses) > 1 and abs(losses[-2] - losses[-1]) < tol:
            break
    return w, losses


# ---------------------------------------------------------------------------
# 10. SMOTE
# ---------------------------------------------------------------------------


def smote(
    X_minority: np.ndarray,
    n_synthetic: int,
    K: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    np.random.seed(random_state)
    n_samples = X_minority.shape[0]
    synthetic_samples: List[np.ndarray] = []
    for _ in range(n_synthetic):
        idx = np.random.randint(0, n_samples)
        x = X_minority[idx]
        distances = [euclidean_distance(x, xi) for xi in X_minority]
        distances[idx] = np.inf
        k_indices = np.argsort(distances)[:K]
        neighbor_idx = int(np.random.choice(k_indices))
        xn = X_minority[neighbor_idx]
        lam = float(np.random.uniform(0, 1))
        synthetic_samples.append(x + lam * (xn - x))
    return np.array(synthetic_samples)


def smote_n_new(
    X_minority: np.ndarray,
    n_new_samples: int,
    k: int = 5,
) -> np.ndarray:
    """与 algorithms/10 接口一致：不固定全局随机种子。"""
    n_samples, n_features = X_minority.shape
    k_eff = min(k, n_samples - 1)
    synthetic = np.zeros((n_new_samples, n_features))
    for i in range(n_new_samples):
        idx = np.random.randint(0, n_samples)
        sample = X_minority[idx]
        distances = np.linalg.norm(X_minority - sample, axis=1)
        neighbor_indices = np.argsort(distances)[1 : k_eff + 1]
        nn_idx = neighbor_indices[np.random.randint(0, len(neighbor_indices))]
        neighbor = X_minority[nn_idx]
        lam = np.random.random()
        synthetic[i] = sample + lam * (neighbor - sample)
    return synthetic


# ---------------------------------------------------------------------------
# examples 自检用：矩阵 PageRank、边表二分图、线性回归 GD（含偏置列）
# ---------------------------------------------------------------------------


def pagerank_power_iteration(
    adj: Dict[int, List[int]],
    n_nodes: int,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
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


def is_bipartite_edges(n: int, edges: List[Tuple[int, int]]) -> bool:
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
                    color[v] = 1 - int(color[u])  # type: ignore[arg-type]
                    q.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def kruskal_mst(
    n: int,
    edges: List[Tuple[int, int, float]],
) -> List[Tuple[int, int, float]]:
    mst, _ = kruskal(n, edges)
    return mst


def linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 500,
) -> np.ndarray:
    m, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        pred = X @ w
        grad = (2.0 / m) * (X.T @ (pred - y))
        w -= lr * grad
    return w


def smote_reference(
    X_minority: np.ndarray,
    k: int = 3,
    n_synthetic: int = 10,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = X_minority.shape
    if n < 2:
        raise ValueError("SMOTE 需要至少 2 个少数类样本")
    k_eff = min(k, n - 1)
    synth = np.zeros((n_synthetic, d))
    for t in range(n_synthetic):
        i = int(rng.integers(0, n))
        dists = np.linalg.norm(X_minority - X_minority[i], axis=1)
        dists[i] = np.inf
        nn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        j = int(rng.choice(nn_idx))
        lam = float(rng.random())
        synth[t] = X_minority[i] + lam * (X_minority[j] - X_minority[i])
    return synth


def information_gain_binary_split(y: np.ndarray, left_mask: np.ndarray) -> float:
    """单阈值划分的信息增益（examples 决策桩用）。"""
    n = len(y)
    if n == 0:
        return 0.0
    parent_h = entropy(y)
    n_l, n_r = int(left_mask.sum()), int((~left_mask).sum())
    if n_l == 0 or n_r == 0:
        return 0.0
    h_l = entropy(y[left_mask])
    h_r = entropy(y[~left_mask])
    return parent_h - (n_l / n) * h_l - (n_r / n) * h_r


def knn_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x: np.ndarray,
    k: int,
) -> int:
    return knn_predict_one(X_train, y_train, x, k)


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
                g = information_gain_binary_split(y, left)
                if g > best_gain:
                    best_gain = g
                    self.feature_idx = j
                    self.threshold = float(t)
                    self.label_left = self._majority(y[left])
                    self.label_right = self._majority(y[~left])

    def predict_row(self, x: np.ndarray) -> int:
        return (
            self.label_left
            if x[self.feature_idx] <= self.threshold
            else self.label_right
        )
