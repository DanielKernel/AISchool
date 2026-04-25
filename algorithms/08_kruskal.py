"""
Kruskal 最小生成树 — 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.UnionFind / kruskal_weight_first
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from aischool.algorithm_core import UnionFind, kruskal_weight_first


def kruskal(n, edges):
    return kruskal_weight_first(n, edges)


if __name__ == "__main__":
    edges = [
        (1, 0, 1),
        (3, 0, 3),
        (4, 1, 2),
        (2, 1, 3),
        (5, 2, 3),
        (7, 2, 4),
        (6, 3, 4),
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
