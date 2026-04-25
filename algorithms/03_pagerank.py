"""
PageRank 算法（幂迭代法）
========================

【算法背景】
PageRank 由 Google 创始人 Larry Page 和 Sergey Brin 于 1998 年提出，
是 Google 搜索引擎最初的核心排名算法。
核心思想："被更多重要网页链接的网页更重要" — 一种递归的重要性定义。

可以想象一个"随机冲浪者"在网页间随机点击链接：
  - 以概率 d (阻尼系数) 点击当前页面上的某个链接
  - 以概率 (1-d) 随机跳转到任意页面
  最终每个页面被访问的长期概率就是它的 PageRank。

【适用场景】
- 搜索引擎排名、社交网络影响力分析、推荐系统、学术论文引用排名

【算法流程（3 步）】
    ┌──────────────────────────────┐
    │ Step 1: 构建转移矩阵 M          │
    │   邻接矩阵行归一化后转置         │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: 初始化排名向量 r        │
    │   r = [1/n, 1/n, ..., 1/n]   │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 3: 幂迭代               │
    │   r_new = (1-d)/n + d * M @ r │◄──┐
    │   收敛? → 结束               │   │
    │   未收敛 → 继续迭代           │───┘
    └──────────────────────────────┘

【核心公式】
  PR(i) = (1-d)/n + d * Σⱼ PR(j)/out(j)

  其中 j 是所有链接到 i 的页面，out(j) 是 j 的出链数。
  向量化表示: r = (1-d)/n + d * M @ r

【口诀】构建转移矩阵 → 幂迭代 → 归一化

【记忆要点 - 考试必背】
  1. 转移矩阵构建:
     - out_degree = adj.sum(axis=1)    → 每个节点的出度
     - M = (adj / out_degree).T        → 行归一化后转置
     ★ 关键理解：adj[i][j]=1 表示 i→j，行归一化后 adj[i][j]=1/out(i)
       转置后 M[j][i]=1/out(i)，这样 M @ r 就是 Σ r[i]/out(i) for i→j
  2. 阻尼系数 d = 0.85:
     - (1-d)/n = 0.15/n 是随机跳转的概率
     - d * M @ r 是沿链接传播的概率
  3. 迭代公式: r_new = (1-d)/n + d * M @ r
     ★ 这一行是整个算法的核心，必须能默写
  4. 收敛判断: np.linalg.norm(r_new - r) < tol

【易错点】
  - 转移矩阵的方向：行归一化 vs 列归一化 → 要转置！
  - 悬挂节点（没有出链的节点）：出度为 0 时要特殊处理，避免除零
  - 阻尼系数的位置：(1-d)/n 加在前面，d 乘在 M@r 前面
  - 初始向量要归一化：所有元素之和为 1

【复杂度】
  时间: O(n² * T)   每次迭代做矩阵-向量乘法 O(n²)，迭代 T 次
  空间: O(n²)       存储转移矩阵
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

    # Step 1: 构建转移矩阵（行归一化后转置）
    out_degree = adj_matrix.sum(axis=1, keepdims=True)
    out_degree[out_degree == 0] = 1   # 避免除零
    M = (adj_matrix / out_degree).T   # 转置使得 M @ r 正确传播

    # Step 2: 初始化排名向量（均匀分布）
    ranks = np.ones(n) / n

    # Step 3: 幂迭代
    for _ in range(max_iters):
        new_ranks = (1 - damping) / n + damping * M @ ranks

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
