"""
图算法核心实现
涵盖：PageRank（幂迭代）、二分图判断（BFS染色）、最小生成树（Kruskal + 并查集）
每个算法仅用标准库，不依赖第三方图库，方便考试手写理解。
"""

from collections import deque


# ============================================================
# 1. PageRank（幂迭代）
# ============================================================
# 原理：
#   PR(u) = (1-d)/N + d * Σ [ PR(v) / OutDegree(v) ]
#   其中求和遍历所有指向 u 的节点 v
#   反复更新所有节点的 PR 值，直到收敛（变化量 < epsilon）
# 参数：阻尼系数 d=0.85，随机跳转概率 (1-d)=0.15
# ============================================================

def pagerank(graph, d=0.85, max_iter=100, epsilon=1e-6):
    """
    PageRank 算法（幂迭代）

    参数：
        graph    : dict，邻接表，graph[u] = [v1, v2, ...]
                   表示节点 u 有出链指向 v1, v2...
        d        : 阻尼系数，通常取 0.85
        max_iter : 最大迭代次数
        epsilon  : 收敛阈值

    返回：
        rank : dict，每个节点的 PageRank 值（已归一化）
    """
    nodes = list(graph.keys())
    N = len(nodes)

    # 初始化：每个节点 PR 值均为 1/N
    rank = {node: 1.0 / N for node in nodes}

    # 构建反向邻接表：in_links[u] = 所有指向 u 的节点
    in_links = {node: [] for node in nodes}
    out_degree = {node: len(graph[node]) for node in nodes}
    for node in nodes:
        for neighbor in graph[node]:
            in_links[neighbor].append(node)

    for iteration in range(max_iter):
        new_rank = {}
        for node in nodes:
            # 随机跳转贡献
            teleport = (1 - d) / N
            # 入链贡献之和
            link_sum = sum(
                rank[src] / out_degree[src]
                for src in in_links[node]
                if out_degree[src] > 0
            )
            new_rank[node] = teleport + d * link_sum

        # 收敛检测：所有节点的 PR 变化量之和 < epsilon
        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank

        if delta < epsilon:
            print(f"  PageRank 第 {iteration+1} 次迭代后收敛（delta={delta:.2e}）")
            break

    return rank


def demo_pagerank():
    print("=" * 50)
    print("【1. PageRank 演示】")
    # 简单有向图：A→B, A→C, B→C, C→A
    graph = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": ["A"],
    }
    rank = pagerank(graph, d=0.85)
    print("  节点 PageRank 值：")
    for node, score in sorted(rank.items(), key=lambda x: -x[1]):
        print(f"    {node}: {score:.6f}")
    print("  （C 被 A、B 都指向，排名应最高）")


# ============================================================
# 2. 二分图判断（BFS 染色法）
# ============================================================
# 原理：
#   从任意未染色节点出发做 BFS，相邻节点交替染色（0/1）
#   若发现相邻节点颜色相同，说明存在奇数环，不是二分图
# 注意：要处理非连通图（对每个连通分量都检测）
# ============================================================

def is_bipartite(graph):
    """
    判断无向图是否是二分图（BFS 染色法）

    参数：
        graph : dict，无向图邻接表，graph[u] = [v1, v2, ...]

    返回：
        (bool, color_dict)：是否二分图，以及各节点的颜色（0 或 1）
    """
    color = {}  # color[node] = 0 或 1

    for start in graph:
        if start in color:
            continue  # 已处理过该连通分量

        # BFS 遍历该连通分量
        queue = deque([start])
        color[start] = 0  # 起始节点染色为 0

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    # 邻居未染色：染与当前节点相反的颜色
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    # 邻居已染色且与当前节点同色：发现冲突，不是二分图
                    return False, color

    return True, color


def demo_bipartite():
    print("=" * 50)
    print("【2. 二分图判断（BFS染色）演示】")

    # 示例1：是二分图（偶数环，无向图）
    bipartite_graph = {
        1: [2, 4],
        2: [1, 3],
        3: [2, 4],
        4: [3, 1],
    }
    result, colors = is_bipartite(bipartite_graph)
    print(f"  图1（正方形 1-2-3-4）: 是二分图？ {result}")
    print(f"  颜色分配: {colors}")

    # 示例2：不是二分图（奇数环：三角形）
    non_bipartite_graph = {
        1: [2, 3],
        2: [1, 3],
        3: [2, 1],
    }
    result2, _ = is_bipartite(non_bipartite_graph)
    print(f"  图2（三角形 1-2-3）: 是二分图？ {result2}")


# ============================================================
# 3. 最小生成树——Kruskal 算法
# ============================================================
# 原理：
#   按边权从小到大排序，贪心地选边
#   用并查集（Union-Find）判断加入这条边是否会成环
#   选了 N-1 条边后停止，得到最小生成树
# 关键：并查集需要路径压缩 + 按秩合并
# ============================================================

class UnionFind:
    """
    并查集（路径压缩 + 按秩合并）
    考试必背：find 和 union 两个方法
    """
    def __init__(self, n):
        self.parent = list(range(n))  # 初始时每个节点的父节点是自身
        self.rank = [0] * n           # 树的高度（秩）

    def find(self, x):
        """找根节点，并路径压缩"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        """合并两个集合，按秩合并；返回是否合并成功（False 表示已在同一集合）"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # 已在同一集合，加入此边会成环
        # 按秩合并：小树挂到大树下
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def kruskal(n, edges):
    """
    Kruskal 最小生成树

    参数：
        n     : 节点数（节点编号 0 到 n-1）
        edges : 边列表，每条边为 (u, v, weight)

    返回：
        mst_edges    : 最小生成树的边列表
        total_weight : 最小生成树的总权重
    """
    # Step 1：按边权从小到大排序
    sorted_edges = sorted(edges, key=lambda e: e[2])

    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0

    for u, v, w in sorted_edges:
        # Step 2：尝试加入这条边（用并查集判断是否成环）
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total_weight += w
            # Step 3：已选 n-1 条边，生成树完成
            if len(mst_edges) == n - 1:
                break

    return mst_edges, total_weight


def demo_kruskal():
    print("=" * 50)
    print("【3. Kruskal 最小生成树演示】")
    # 5 个节点，7 条边
    # 节点编号 0-4
    edges = [
        (0, 1, 4),
        (0, 2, 3),
        (1, 2, 1),
        (1, 3, 2),
        (2, 3, 4),
        (3, 4, 2),
        (2, 4, 5),
    ]
    n = 5
    mst, total = kruskal(n, edges)
    print(f"  图: {n} 个节点, {len(edges)} 条边")
    print(f"  最小生成树的边:")
    for u, v, w in mst:
        print(f"    {u} --({w})-- {v}")
    print(f"  最小生成树总权重: {total}")
    print(f"  （最优解：1+2+2+3=8）")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("\n==================================================")
    print("   图算法演示（PageRank / 二分图 / Kruskal MST）")
    print("==================================================\n")

    demo_pagerank()
    print()
    demo_bipartite()
    print()
    demo_kruskal()
    print()
    print("所有图算法演示完毕！")
