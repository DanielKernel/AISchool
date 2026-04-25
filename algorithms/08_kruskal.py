"""
Kruskal 最小生成树算法
=====================
核心思想：将所有边按权重排序 → 逐边加入，用并查集判断是否形成环 → 直到选够 n-1 条边

口诀：排序边 → 并查集 → 逐边加入（不成环就加）
"""


class UnionFind:
    """并查集（带路径压缩和按秩合并）"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """查找根节点（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """合并两个集合，返回是否成功合并（已在同一集合则返回 False）"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # 已在同一集合（会形成环）
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

    # Step 3: 逐边加入
    for weight, u, v in edges_sorted:
        if uf.union(u, v):  # 不形成环就加入
            mst_edges.append((u, v, weight))
            total_weight += weight
            if len(mst_edges) == n - 1:  # 已选够 n-1 条边
                break

    return mst_edges, total_weight


# ========== 运行示例 ==========
if __name__ == "__main__":
    # 5 个节点的图
    #     1
    #    /|\
    #   4 | 2
    #  /  |  \
    # 0---3---4
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
    print("最小生成树的边:")
    for u, v, w in mst:
        print(f"  {u} -- {v}, 权重: {w}")
    print(f"总权重: {total}")
