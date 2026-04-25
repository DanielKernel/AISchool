"""
二分图判断（BFS 染色法）
=======================

【算法背景】
二分图 (Bipartite Graph) 是指能将所有节点分为两组 (U, V)，
使得每条边都连接 U 中的节点和 V 中的节点（即同组内没有边）。

判断方法：用 BFS 交替染两种颜色，如果某条边两端颜色相同则不是二分图。
直觉：就像给地图上的国家涂色，相邻国家不能同色 — 能用 2 种颜色完成吗？

等价命题: 图是二分图 ⟺ 图中不包含奇数长度的环

【适用场景】
- 社交网络中的敌友关系建模（"敌人的敌人是朋友"）
- 任务分配问题（人 vs 任务）
- 匹配问题的前提检查

【算法流程】
    ┌──────────────────────────────┐
    │ 对图的每个连通分量:            │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 1: 选一个未染色的起始节点  │
    │   染色为 0，加入 BFS 队列      │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: BFS 遍历              │
    │  从队列取出节点 u              │◄──┐
    │  for 每个邻居 v:               │   │
    │    未染色 → 染 1-color[u]      │   │
    │             加入队列           │   │
    │    已染色且同色 → return False  │   │
    │  队列非空 → 继续               │───┘
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ 所有节点处理完 → return True   │
    └──────────────────────────────┘

【口诀】BFS → 染色 → 检查冲突

【记忆要点 - 考试必背】
  1. 数据结构: color = {} 字典记录每个节点的颜色 (0 或 1)
     - 用 dict 而不是 list，因为可以用 `in` 检查是否已染色
  2. 染色技巧: color[neighbor] = 1 - color[node]
     ★ "1 减当前色" 实现交替染色（0↔1）
  3. 冲突检测: color[neighbor] == color[node] → return False
     ★ 相邻同色 = 矛盾 = 不是二分图
  4. 多连通分量: for start in graph: 遍历所有节点
     ★ 图可能不连通！每个连通分量都要检查

【关键代码模式 (BFS 模板)】
    queue = deque([start])
    color[start] = 0
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in color:
                color[neighbor] = 1 - color[node]
                queue.append(neighbor)
            elif color[neighbor] == color[node]:
                return False

【典型例子】
  偶数环 (4节点): 0-1-2-3-0 → 是二分图 ✓
    0(红)-1(蓝)-2(红)-3(蓝) → 无冲突

  奇数环 (三角形): 0-1-2-0 → 不是二分图 ✗
    0(红)-1(蓝)-2(红)-0(红) → 2和0都是红但相邻！

  完全二分图 K_{2,3}: → 是二分图 ✓
    {0,1}(红) ↔ {2,3,4}(蓝)

【易错点】
  - 图可能有多个连通分量 → 外层循环遍历所有节点
  - 用 BFS 不用 DFS 更好理解（虽然 DFS 也可以）
  - 邻接表要双向：如果 0-1 相连，graph[0] 包含 1，graph[1] 也要包含 0
  - color 用 dict 不用 list，更灵活

【复杂度】
  时间: O(V + E)  遍历所有节点和边
  空间: O(V)      color 字典 + BFS 队列
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
                    color[neighbor] = 1 - color[node]  # 交替染色
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False  # 冲突！

    return True


# ========== 运行示例 ==========
if __name__ == "__main__":
    print("二分图判断 (BFS 染色法):")
    print("=" * 50)

    # 例 1: 偶数环 → 二分图
    graph1 = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [2, 0]}
    result1 = is_bipartite(graph1)
    print(f"图1 (4节点偶数环 0-1-2-3-0): {result1}")
    print(f"  染色方案: 0→红, 1→蓝, 2→红, 3→蓝 → 无冲突 ✓")

    print()

    # 例 2: 三角形 → 非二分图
    graph2 = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    result2 = is_bipartite(graph2)
    print(f"图2 (三角形 0-1-2-0): {result2}")
    print(f"  0→红, 1→蓝, 2→? 与0相邻应蓝，与1相邻应红 → 矛盾 ✗")

    print()

    # 例 3: 完全二分图
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
