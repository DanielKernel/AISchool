"""
二分图判断（BFS 染色法）— 详解见 STUDY_GUIDE.md。
核心实现：aischool.algorithm_core.is_bipartite
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from collections import deque

from aischool.algorithm_core import is_bipartite as _is_bipartite_core


def is_bipartite(graph):
    ok, _ = _is_bipartite_core(graph)
    return ok


if __name__ == "__main__":
    print("二分图判断 (BFS 染色法):")
    print("=" * 50)

    graph1 = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [2, 0]}
    result1 = is_bipartite(graph1)
    print(f"图1 (4节点偶数环 0-1-2-3-0): {result1}")
    print(f"  染色方案: 0→红, 1→蓝, 2→红, 3→蓝 → 无冲突 ✓")

    print()

    graph2 = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    result2 = is_bipartite(graph2)
    print(f"图2 (三角形 0-1-2-0): {result2}")
    print(f"  0→红, 1→蓝, 2→? 与0相邻应蓝，与1相邻应红 → 矛盾 ✗")

    print()

    graph3 = {0: [2, 3, 4], 1: [2, 3, 4], 2: [0, 1], 3: [0, 1], 4: [0, 1]}
    result3 = is_bipartite(graph3)
    print(f"图3 (完全二分图 K_2,3): {result3}")
    print(f"  U={{0,1}}→红, V={{2,3,4}}→蓝 → 无冲突 ✓")

    print()
    print("--- 理解辅助：BFS 染色过程追踪 (图1) ---")
    color_trace = {}
    queue = deque([0])
    color_trace[0] = 0
    step = 0
    while queue:
        node = queue.popleft()
        for neighbor in graph1[node]:
            if neighbor not in color_trace:
                color_trace[neighbor] = 1 - color_trace[node]
                queue.append(neighbor)
                step += 1
                c_name = "红" if color_trace[neighbor] == 0 else "蓝"
                print(f"  Step {step}: 节点 {node}({'红' if color_trace[node]==0 else '蓝'}) "
                      f"→ 邻居 {neighbor} 染{c_name}")
