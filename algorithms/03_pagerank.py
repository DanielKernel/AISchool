"""
PageRank 算法（幂迭代法）
========================
核心思想：构建转移矩阵 M → 反复用 M 乘以当前排名向量 → 直到向量收敛

口诀：构建转移矩阵 → 幂迭代 → 归一化
"""
import numpy as np


def pagerank(adj_matrix, damping=0.85, max_iters=100, tol=1e-6):
    """
    参数:
        adj_matrix: 邻接矩阵, shape (n, n), adj[i][j]=1 表示 i→j 有链接
        damping: 阻尼系数 (通常 0.85)
        max_iters: 最大迭代次数
        tol: 收敛阈值
    返回:
        ranks: 每个节点的 PageRank 值, shape (n,)
    """
    n = adj_matrix.shape[0]

    # Step 1: 构建转移矩阵（列归一化）
    out_degree = adj_matrix.sum(axis=1, keepdims=True)
    # 避免除零：没有出链的节点均匀分配
    out_degree[out_degree == 0] = 1
    M = (adj_matrix / out_degree).T  # 转置使得 M 的每列和为 1

    # Step 2: 初始化排名向量（均匀分布）
    ranks = np.ones(n) / n

    # Step 3: 幂迭代
    for _ in range(max_iters):
        new_ranks = (1 - damping) / n + damping * M @ ranks

        # 检查收敛
        if np.linalg.norm(new_ranks - ranks) < tol:
            break
        ranks = new_ranks

    return ranks


# ========== 运行示例 ==========
if __name__ == "__main__":
    # 4 个网页的链接关系:  0→1, 0→2, 1→2, 2→0, 3→2
    adj = np.array([
        [0, 1, 1, 0],  # 页面0 链接到 1, 2
        [0, 0, 1, 0],  # 页面1 链接到 2
        [1, 0, 0, 0],  # 页面2 链接到 0
        [0, 0, 1, 0],  # 页面3 链接到 2
    ], dtype=float)

    ranks = pagerank(adj)
    for i, r in enumerate(ranks):
        print(f"页面 {i}: PageRank = {r:.4f}")
