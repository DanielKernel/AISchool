"""
PageRank 算法（幂迭代法）— 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.pagerank_adjacency
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from aischool.algorithm_core import pagerank_adjacency


def pagerank(adj_matrix, damping=0.85, max_iters=100, tol=1e-6):
    return pagerank_adjacency(adj_matrix, damping=damping, max_iters=max_iters, tol=tol)


if __name__ == "__main__":
    adj = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ], dtype=float)

    ranks = pagerank(adj)
    print("PageRank 结果:")
    print("=" * 40)
    for i, r in enumerate(ranks):
        print(f"  页面 {i}: PageRank = {r:.4f}")

    print()
    print("--- 理解辅助：为什么页面 2 排名最高？ ---")
    print("  页面 2 被页面 0、1、3 链接 (入度=3)，收到最多'投票'")
    print("  页面 0 被页面 2 链接 (入度=1)，但页面 2 的 PR 值高，所以页面 0 也不低")
    print("  页面 3 没有被任何页面链接 (入度=0)，只有随机跳转的基础分")

    print()
    print("--- 理解辅助：手动验证一次迭代 ---")
    n = 4
    d = 0.85
    out_deg = adj.sum(axis=1, keepdims=True)
    out_deg[out_deg == 0] = 1
    M = (adj / out_deg).T
    r = np.ones(n) / n
    r_new = (1 - d) / n + d * M @ r
    print(f"  初始 r = {r}")
    print(f"  一次迭代后 r = [{', '.join(f'{x:.4f}' for x in r_new)}]")
