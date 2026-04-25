"""
二分图判断（BFS 染色法）
=======================
核心思想：从任意节点开始 BFS → 交替染两种颜色 → 如果相邻节点同色则不是二分图

口诀：BFS → 染色 → 检查冲突
"""
from collections import deque


def is_bipartite(graph):
    """
    参数:
        graph: 邻接表, dict[int, list[int]]
               graph[u] = [v1, v2, ...] 表示 u 与 v1, v2, ... 相连
    返回:
        bool: 是否为二分图
    """
    color = {}  # 节点 → 颜色 (0 或 1)

    # 遍历所有连通分量
    for start_node in graph:
        if start_node in color:
            continue

        # BFS 染色
        queue = deque([start_node])
        color[start_node] = 0

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    # 相邻节点染相反的颜色
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    # 冲突！相邻节点同色 → 不是二分图
                    return False

    return True


# ========== 运行示例 ==========
if __name__ == "__main__":
    # 二分图: 0-1-2-3 形成环（偶数环）
    graph1 = {
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [2, 0],
    }
    print(f"图1 (4节点偶数环): 是二分图? {is_bipartite(graph1)}")  # True

    # 非二分图: 0-1-2 三角形（奇数环）
    graph2 = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }
    print(f"图2 (三角形):       是二分图? {is_bipartite(graph2)}")  # False

    # 二分图: 完全二分图 K_{2,3}
    graph3 = {
        0: [2, 3, 4],
        1: [2, 3, 4],
        2: [0, 1],
        3: [0, 1],
        4: [0, 1],
    }
    print(f"图3 (K_2,3):       是二分图? {is_bipartite(graph3)}")  # True
