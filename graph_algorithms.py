"""
图算法演示（实现见 aischool.algorithm_core）。
涵盖：PageRank（幂迭代）、二分图判断（BFS染色）、最小生成树（Kruskal + 并查集）
"""

from aischool.algorithm_core import UnionFind, is_bipartite, kruskal, pagerank


def demo_pagerank():
    print("=" * 50)
    print("【1. PageRank 演示】")
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


def demo_bipartite():
    print("=" * 50)
    print("【2. 二分图判断（BFS染色）演示】")

    bipartite_graph = {
        1: [2, 4],
        2: [1, 3],
        3: [2, 4],
        4: [3, 1],
    }
    result, colors = is_bipartite(bipartite_graph)
    print(f"  图1（正方形 1-2-3-4）: 是二分图？ {result}")
    print(f"  颜色分配: {colors}")

    non_bipartite_graph = {
        1: [2, 3],
        2: [1, 3],
        3: [2, 1],
    }
    result2, _ = is_bipartite(non_bipartite_graph)
    print(f"  图2（三角形 1-2-3）: 是二分图？ {result2}")


def demo_kruskal():
    print("=" * 50)
    print("【3. Kruskal 最小生成树演示】")
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


# 供外部 from graph_algorithms import UnionFind 等
__all__ = ["UnionFind", "pagerank", "is_bipartite", "kruskal", "demo_pagerank", "demo_bipartite", "demo_kruskal"]


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
