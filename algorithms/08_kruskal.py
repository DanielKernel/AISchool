"""
Kruskal 最小生成树算法
=====================

【算法背景】
最小生成树 (MST) 问题：给定一个带权无向连通图，找到一棵树，
包含图中所有节点，且总边权最小。

Kruskal 算法由 Joseph Kruskal 于 1956 年提出，采用贪心策略：
"从最短的边开始，逐条加入，不成环就加"。

【适用场景】
- 网络布线（最小成本连接所有节点）、道路规划、集群分析

【核心数据结构 —— 并查集 (Union-Find)】
  并查集用于高效判断两个节点是否在同一集合（是否已连通）。
  两个核心操作：
    find(x):     找 x 所在集合的根节点
    union(x, y): 合并 x 和 y 所在的集合

  优化技巧（必须掌握）：
    路径压缩: find 时把沿途节点直接挂到根节点上
    按秩合并: 矮的树挂到高的树上，避免退化为链表

  并查集代码模板:
    find(x): if parent[x] != x: parent[x] = find(parent[x]); return parent[x]
    union(x,y): rx, ry = find(x), find(y); if rx==ry: return False; parent[ry]=rx

【算法流程】
    ┌──────────────────────────────┐
    │ Step 1: 将所有边按权重从小到大  │
    │   排序                        │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: 初始化并查集           │
    │   每个节点是自己的根           │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 3: 依次取最小边 (u, v)   │
    │   find(u) == find(v)?        │◄──┐
    │   是 → 跳过（会成环）         │   │
    │   否 → 加入 MST, union(u,v)  │   │
    │   已选 n-1 条边? → 结束       │   │
    │   否 → 继续                   │───┘
    └──────────────────────────────┘

【口诀】排序边 → 并查集 → 逐边加入（不成环就加）

【记忆要点 - 考试必背】
  1. 并查集 find (路径压缩):
     def find(x):
         if parent[x] != x:
             parent[x] = find(parent[x])   ← 递归时顺便压缩路径
         return parent[x]

  2. 并查集 union:
     rx, ry = find(x), find(y)
     if rx == ry: return False   ← 已在同一集合 = 会形成环
     parent[ry] = rx; return True

  3. Kruskal 主逻辑:
     edges.sort()
     for w, u, v in edges:
         if union(u, v):         ← 不成环就加入
             mst.append((u,v,w))
         if len(mst) == n-1:     ← 树有 n-1 条边
             break

  4. MST 一定有 n-1 条边（n 是节点数）

【Kruskal vs Prim（常考对比）】
  Kruskal: 按边排序，贪心选边 → 适合稀疏图 O(E log E)
  Prim:    从一个点出发，贪心扩展 → 适合稠密图 O(V²) 或 O(E log V)

【易错点】
  - 边的格式: (weight, u, v) — 权重放前面方便排序
  - 路径压缩是 parent[x] = find(parent[x])，不是 parent[x] = parent[parent[x]]
  - 终止条件: len(mst) == n-1，不是 n
  - 初始化: parent[i] = i，每个节点的父亲是自己

【复杂度】
  时间: O(E log E)  排序边的时间（并查集操作近似 O(1)）
  空间: O(V)        并查集数组
"""


class UnionFind:
    """并查集（带路径压缩和按秩合并）"""

    def __init__(self, n):
        self.parent = list(range(n))  # parent[i] = i 初始化
        self.rank = [0] * n

    def find(self, x):
        """查找根节点（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        """合并两个集合，返回是否成功（已在同一集合返回 False = 会成环）"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def kruskal(n, edges):
    """
    参数:
        n:     节点数量 (节点编号 0 到 n-1)
        edges: 边列表, [(weight, u, v), ...]
    返回:
        mst_edges:    最小生成树的边列表
        total_weight: 总权重
    """
    # Step 1: 按权重排序所有边
    edges_sorted = sorted(edges, key=lambda e: e[0])

    # Step 2: 初始化并查集
    uf = UnionFind(n)

    mst_edges = []
    total_weight = 0

    # Step 3: 贪心逐边加入
    for weight, u, v in edges_sorted:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            total_weight += weight
            if len(mst_edges) == n - 1:
                break

    return mst_edges, total_weight


# ========== 运行示例 ==========
if __name__ == "__main__":
    edges = [
        (1, 0, 1),  # 0-1, 权重1
        (3, 0, 3),  # 0-3, 权重3
        (4, 1, 2),  # 1-2, 权重4
        (2, 1, 3),  # 1-3, 权重2
        (5, 2, 3),  # 2-3, 权重5
        (7, 2, 4),  # 2-4, 权重7
        (6, 3, 4),  # 3-4, 权重6
    ]

    mst, total = kruskal(5, edges)
    print("Kruskal 最小生成树:")
    print("=" * 40)
    print("最小生成树的边:")
    for u, v, w in mst:
        print(f"  {u} -- {v}, 权重: {w}")
    print(f"总权重: {total}")
    print(f"边数: {len(mst)} (应为 n-1 = {5-1})")

    print()
    print("--- 理解辅助：贪心过程追踪 ---")
    edges_sorted = sorted(edges, key=lambda e: e[0])
    uf = UnionFind(5)
    for w, u, v in edges_sorted:
        ru, rv = uf.find(u), uf.find(v)
        if ru != rv:
            uf.union(u, v)
            print(f"  边 {u}-{v} (权重{w}): find({u})={ru}, find({v})={rv} → 不同集合，加入 ✓")
        else:
            print(f"  边 {u}-{v} (权重{w}): find({u})={ru}, find({v})={rv} → 同一集合，跳过 ✗ (会成环)")
