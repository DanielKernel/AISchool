"""
核心算法的唯一实现源（供根目录演示、examples 自检、algorithms/*.py 引用）。
含 KNN/KMeans/PageRank/决策树(信息增益)/卷积与池化/MST/二分图/梯度下降/SMOTE/DTW 等。
依赖：numpy；图算法中二分图与 PageRank（字典版）仅用标准库。
"""

from __future__ import annotations

import sys
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sliding_window_2d(
    x: np.ndarray,
    window_h: int,
    window_w: int,
    stride_h: int,
    stride_w: int,
) -> np.ndarray:
    """形状 (out_h, out_w, window_h, window_w) 的滑动窗口视图（零拷贝）。"""
    x = np.ascontiguousarray(x)
    H, W = x.shape
    out_h = (H - window_h) // stride_h + 1
    out_w = (W - window_w) // stride_w + 1
    s0, s1 = x.strides
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(out_h, out_w, window_h, window_w),
        strides=(stride_h * s0, stride_w * s1, s0, s1),
        writeable=False,
    )


# ---------------------------------------------------------------------------
# 距离
# ---------------------------------------------------------------------------


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离（L2）。"""
    return float(np.linalg.norm(a - b))


def _dtw_xy_to_2d(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """1D 视为长度 T、特征维 1；已是 (T, d) 则保持不变。"""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    if ya.ndim == 1:
        ya = ya.reshape(-1, 1)
    if xa.ndim != 2 or ya.ndim != 2:
        raise ValueError("dtw: x, y 须为 1D 时间序列或 2D (时间步, 特征)")
    if xa.shape[1] != ya.shape[1]:
        raise ValueError("dtw: 特征维 d 须一致")
    return xa, ya


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    squared: bool = True,
) -> float:
    """动态时间规整（DTW）距离。`x` 形状 `(n,)` 或 `(n, d)`，`y` 形状 `(m,)` 或 `(m, d)`。

    代价为逐点平方欧氏和（`squared=True`）或逐点 L2（`squared=False`）。
    返回与经典实现一致的标量距离（平方模式返回的是路径上平方代价之和，非再开方）。
    """
    xa, ya = _dtw_xy_to_2d(x, y)
    n, m = xa.shape[0], ya.shape[0]
    diff = xa[:, None, :] - ya[None, :, :]
    if squared:
        cost = np.sum(diff * diff, axis=2)
    else:
        cost = np.linalg.norm(diff, axis=2)
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = float(cost[i - 1, j - 1])
            dtw[i, j] = c + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def dtw_warping_path(
    x: np.ndarray,
    y: np.ndarray,
    squared: bool = True,
) -> Tuple[float, List[Tuple[int, int]]]:
    """DTW 累计代价与对齐路径（元组为 `x` / `y` 的 0 起始下标；并列时优先对角步）。"""
    xa, ya = _dtw_xy_to_2d(x, y)
    n, m = xa.shape[0], ya.shape[0]
    diff = xa[:, None, :] - ya[None, :, :]
    if squared:
        cost = np.sum(diff * diff, axis=2)
    else:
        cost = np.linalg.norm(diff, axis=2)
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = float(cost[i - 1, j - 1])
            dtw[i, j] = c + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            path.append((0, j))
        elif j == 0:
            i -= 1
            path.append((i, 0))
        else:
            path.append((i - 1, j - 1))
            c = float(cost[i - 1, j - 1])
            prev = dtw[i, j] - c
            diag, up, left = dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1]
            if np.isclose(prev, diag):
                i, j = i - 1, j - 1
            elif np.isclose(prev, up):
                i -= 1
            else:
                j -= 1
    path.reverse()
    return float(dtw[n, m]), path


# ---------------------------------------------------------------------------
# 1. K-Means
# ---------------------------------------------------------------------------


def _kmeans_centers_from_labels(X: np.ndarray, labels: np.ndarray, k: int, old: np.ndarray) -> np.ndarray:
    """按标签聚合并求均值；空簇保留 old 中对应簇心。"""
    d = X.shape[1]
    sums = np.zeros((k, d), dtype=X.dtype)
    np.add.at(sums, labels, X)
    counts = np.bincount(labels, minlength=k).astype(X.dtype)[:, None]
    return np.where(counts > 0, sums / counts, old)


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
        new_centers = _kmeans_centers_from_labels(X, labels, k, centers)
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
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = _kmeans_centers_from_labels(X, new_labels, K, centers)
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
    return knn_predict_batch(X_train, y_train, X_test, k=K)


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
    return int(Counter(k_nearest_labels).most_common(1)[0][0])


def knn_predict_batch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """||x - t||² = ||x||² + ||t||² - 2 x·t，一次矩阵乘算全体测试点到训练集距离。"""
    t_norm = np.sum(X_train * X_train, axis=1)
    x_norm = np.sum(X_test * X_test, axis=1, keepdims=True)
    dist_sq = np.maximum(x_norm + t_norm - 2.0 * (X_test @ X_train.T), 0.0)
    k_eff = min(k, X_train.shape[0])
    # 与逐点 linalg.norm + argsort 的并列次序一致（argpartition 在并列时可能选不同邻居）
    k_nn = np.argsort(dist_sq, axis=1)[:, :k_eff]
    k_labels = y_train[k_nn]
    preds = np.empty(X_test.shape[0], dtype=int)
    for i in range(X_test.shape[0]):
        preds[i] = int(Counter(k_labels[i]).most_common(1)[0][0])
    return preds


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
    p = counts.astype(np.float64) / len(y)
    return float(-np.sum(p * np.log2(np.clip(p, 1e-12, None))))


def information_gain(X_col: np.ndarray, y: np.ndarray) -> float:
    parent_entropy = entropy(y)
    n = len(y)
    if n == 0:
        return 0.0
    _, inv = np.unique(X_col, return_inverse=True)
    g = int(inv.max()) + 1 if inv.size else 0
    sizes = np.bincount(inv, minlength=g)
    weighted = 0.0
    for gid in range(g):
        sz = int(sizes[gid])
        if sz == 0:
            continue
        weighted += (sz / n) * entropy(y[inv == gid])
    return parent_entropy - weighted


def best_split_feature(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    gains = np.array([information_gain(X[:, j], y) for j in range(X.shape[1])])
    j = int(np.argmax(gains))
    return j, float(gains[j])


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
    else:
        input_map = np.asarray(input_map)
    kH, kW = kernel.shape
    patches = _sliding_window_2d(input_map, kH, kW, stride, stride)
    return np.tensordot(patches, kernel, axes=([2, 3], [0, 1]))


def max_pooling2d(
    input_map: np.ndarray,
    pool_size: int = 2,
    stride: Optional[int] = None,
) -> np.ndarray:
    if stride is None:
        stride = pool_size
    patches = _sliding_window_2d(
        input_map, pool_size, pool_size, stride, stride
    )
    return np.max(patches, axis=(2, 3))


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
    patches = _sliding_window_2d(x, pool_h, pool_w, stride, stride)
    return np.max(patches, axis=(2, 3))


def avg_pooling2d(
    input_map: np.ndarray,
    pool_size: int = 2,
    stride: Optional[int] = None,
) -> np.ndarray:
    if stride is None:
        stride = pool_size
    patches = _sliding_window_2d(
        input_map, pool_size, pool_size, stride, stride
    )
    return np.mean(patches, axis=(2, 3))


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
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        pr, rk = self.parent, self.rank
        if rk[rx] < rk[ry]:
            rx, ry = ry, rx
        pr[ry] = rx
        if rk[rx] == rk[ry]:
            rk[rx] += 1
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
    ar = np.asarray(edges, dtype=np.float64)
    ar = ar[np.argsort(ar[:, 0])]
    as_uvw = [(int(r[1]), int(r[2]), float(r[0])) for r in ar]
    return kruskal(n, as_uvw)


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
    early_stop: bool = True,
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
        if (
            early_stop
            and len(losses) > 1
            and abs(losses[-2] - losses[-1]) < tol
        ):
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
    n_samples, n_features = X_minority.shape
    out = np.empty((n_synthetic, n_features), dtype=X_minority.dtype)
    for i in range(n_synthetic):
        idx = np.random.randint(0, n_samples)
        x = X_minority[idx]
        diff = X_minority - x
        dist_sq = np.einsum("ij,ij->i", diff, diff, optimize=True)
        dist_sq[idx] = np.inf
        k_indices = np.argsort(dist_sq)[:K]
        neighbor_idx = int(np.random.choice(k_indices))
        lam = float(np.random.uniform(0, 1))
        out[i] = x + lam * (X_minority[neighbor_idx] - x)
    return out


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
        diff = X_minority - sample
        dist_sq = np.einsum("ij,ij->i", diff, diff, optimize=True)
        neighbor_indices = np.argsort(dist_sq)[1 : k_eff + 1]
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
    w, _ = gradient_descent(X, y, lr=lr, max_iters=epochs, early_stop=False)
    return w


def linear_regression_gd_recursive(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 500,
    w: Optional[np.ndarray] = None,
    depth: int = 0,
    max_depth: Optional[int] = None,
) -> np.ndarray:
    """线性回归 MSE 的批量梯度下降（递归形式，数学上等价于循环版）。

    仅作教学演示：Python 递归深度有限，默认将迭代步数限制为
    ``min(epochs, max_depth, sys.getrecursionlimit() - 80)`` 以防栈溢出；
    长训练请用 `linear_regression_gd` / `gradient_descent`。
    """
    m, d = X.shape
    if w is None:
        w = np.zeros(d)
    lim = sys.getrecursionlimit() - 80
    cap = min(epochs, max_depth if max_depth is not None else epochs, lim)
    if depth >= cap:
        return w
    pred = X @ w
    grad = (2.0 / m) * (X.T @ (pred - y))
    w_new = w - lr * grad
    return linear_regression_gd_recursive(
        X, y, lr=lr, epochs=epochs, w=w_new, depth=depth + 1, max_depth=max_depth
    )


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
        diff = X_minority - X_minority[i]
        dist_sq = np.einsum("ij,ij->i", diff, diff, optimize=True)
        dist_sq[i] = np.inf
        nn_idx = np.argpartition(dist_sq, k_eff - 1)[:k_eff]
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
