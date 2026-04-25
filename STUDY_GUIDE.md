# AI 算法速成学习指南

> 目标：一天之内掌握 10 个核心 AI / 数据科学 / 图论算法的 Python 实现，能够在考试中手写或默写关键代码。

---

## 一、学习策略总览

### 核心原则

| 原则 | 说明 |
|------|------|
| **理解优先于记忆** | 先搞清楚算法「为什么这样做」，再记「怎么写」 |
| **最小可运行代码** | 每个算法只保留最核心的 20-30 行，去掉所有工程化代码 |
| **输入→过程→输出** | 每个算法都用统一框架理解：输入是什么、核心循环做什么、输出是什么 |
| **手写复现** | 看完代码后合上屏幕，在纸上默写，这是最有效的记忆方式 |
| **分组联想** | 把思维模式相似的算法放在一起记忆，互相关联 |

### 算法分组记忆

将 10 个算法按「思维模式」分为 4 组，同组算法有共同的思考框架：

| 组别 | 算法 | 共同模式 | 联想记忆 |
|------|------|----------|----------|
| **迭代收敛** | K-Means、PageRank、梯度下降 | `while not converged: update()` | 像练字——写了擦、擦了写，直到满意 |
| **分类/决策** | KNN、决策树 | 用某种度量来做分类 | KNN 像投票选班长，决策树像玩"20个问题" |
| **矩阵/窗口** | 卷积、池化 | 双层循环滑动窗口 | 像拿放大镜在图片上扫描 |
| **图算法** | 二分图、Kruskal | BFS/并查集处理图 | 二分图像给地图涂两种色，Kruskal 像修最省钱的路 |
| **数据增强** | SMOTE | K 近邻 + 插值 | 像在两个同类样本之间"凭空生成"新样本 |

---

## 二、一天学习计划（时间分配）

### 第一阶段：迭代收敛组（约 2 小时）

| 时间段 | 内容 | 文件 |
|--------|------|------|
| 30 min | **K-Means**：理解「随机初始化→分配→更新簇心」循环 | `algorithms/01_kmeans.py` |
| 30 min | **PageRank**：理解「矩阵乘法做幂迭代」 | `algorithms/03_pagerank.py` |
| 30 min | **梯度下降**：理解「计算梯度→更新参数」循环 | `algorithms/09_gradient_descent.py` |
| 30 min | 合上屏幕，在纸上默写这 3 个算法的核心循环 | — |

**记忆锚点**：这三个算法都是 `while not converged: update()` 的模式。区别只在「更新什么」：
- K-Means 更新**簇心**（均值）
- PageRank 更新**排名向量**（矩阵乘法）
- 梯度下降更新**模型参数**（减去梯度）

### 第二阶段：分类/决策组（约 1.5 小时）

| 时间段 | 内容 | 文件 |
|--------|------|------|
| 30 min | **KNN**：理解「算距离→排序→投票」三步 | `algorithms/02_knn.py` |
| 30 min | **决策树**：理解「计算信息增益→递归分裂」 | `algorithms/04_decision_tree.py` |
| 30 min | 默写练习 | — |

**记忆锚点**：KNN 是「找最近的 K 个邻居投票」，决策树是「找最好的特征切一刀，递归」。两者都要"选最优"，KNN 选最近的点，决策树选最好的特征。

### 第三阶段：矩阵/窗口运算组（约 1 小时）

| 时间段 | 内容 | 文件 |
|--------|------|------|
| 25 min | **卷积**：理解「滑动窗口 × 卷积核 → 求和」 | `algorithms/05_convolution.py` |
| 25 min | **池化**：理解「滑动窗口 → 取最大值」 | `algorithms/06_max_pooling.py` |
| 10 min | 默写练习 | — |

**记忆锚点**：两个算法代码框架**几乎一模一样**，只有窗口内的操作不同：
```
卷积: output[i,j] = np.sum(region * kernel)   ← 加权求和
池化: output[i,j] = np.max(region)            ← 取最大值
```

### 第四阶段：图算法组（约 1.5 小时）

| 时间段 | 内容 | 文件 |
|--------|------|------|
| 40 min | **二分图判断**：理解「BFS + 染色，相邻不同色」 | `algorithms/07_bipartite_check.py` |
| 40 min | **Kruskal 最小生成树**：理解「排序边 + 并查集判环」 | `algorithms/08_kruskal.py` |
| 10 min | 默写练习 | — |

**记忆锚点**：二分图用 BFS 队列 + 颜色字典，Kruskal 用排序 + 并查集。两者都在处理图结构，但目的不同：判断性质 vs 构建最优子图。

### 第五阶段：数据增强 + 总复习（约 2 小时）

| 时间段 | 内容 | 文件 |
|--------|------|------|
| 30 min | **SMOTE**：理解「找 K 近邻 → 随机插值」 | `algorithms/10_smote.py` |
| 90 min | **全部 10 个算法默写测试**（每个限时 8 分钟） | — |

---

## 三、每个算法详细记忆卡片

### 3.1 K-Means 聚类

| 项目 | 内容 |
|------|------|
| **一句话** | 随机选 K 个簇心，反复「分配→更新」直到不动 |
| **口诀** | 初始化→分配→更新→重复 |
| **类型** | 无监督学习 / 聚类 |
| **输入** | 数据矩阵 X (n×d), 簇数 K |
| **输出** | K 个簇心坐标, 每个样本的标签 |

**必背代码骨架：**
```python
centroids = X[random_indices]         # 初始化
for _ in range(max_iters):
    distances = norm(X - centroids)   # 算距离
    labels = argmin(distances)        # 分配
    centroids = mean(X[labels==i])    # 更新
    if 不变了: break                  # 收敛
```

**关键 NumPy 技巧：**
- `X[:, np.newaxis]` — 升维实现广播
- `np.linalg.norm(..., axis=2)` — 批量求距离
- `X[labels == i].mean(axis=0)` — 布尔索引 + 求均值

---

### 3.2 KNN 分类

| 项目 | 内容 |
|------|------|
| **一句话** | 算距离、找最近 K 个、投票 |
| **口诀** | 算距离→排序→投票 |
| **类型** | 有监督学习 / 分类（懒惰学习） |
| **输入** | 训练集 (X_train, y_train), 查询点 x, K 值 |
| **输出** | 预测的类别标签 |

**必背代码骨架：**
```python
distances = norm(X_train - x_query)   # 算距离
k_idx = argsort(distances)[:k]        # 排序取前K
labels = y_train[k_idx]              # 取标签
return Counter(labels).most_common(1) # 投票
```

**易混淆对比：KNN vs K-Means**
| | KNN | K-Means |
|---|------|---------|
| 类型 | 有监督/分类 | 无监督/聚类 |
| K 含义 | 近邻数 | 簇数 |
| 训练 | 无（懒惰学习）| 迭代收敛 |

---

### 3.3 PageRank

| 项目 | 内容 |
|------|------|
| **一句话** | 构建转移矩阵，反复矩阵乘法直到排名稳定 |
| **口诀** | 构建转移矩阵→幂迭代→归一化 |
| **类型** | 图算法 / 链接分析 |
| **输入** | 邻接矩阵, 阻尼系数 d=0.85 |
| **输出** | 每个节点的排名分数 |

**必背代码骨架：**
```python
M = (adj / out_degree).T             # 转移矩阵
r = ones(n) / n                      # 初始排名
for _ in range(max_iters):
    r_new = (1-d)/n + d * M @ r      # ★ 核心公式
    if 收敛: break
    r = r_new
```

**关键记忆：** `r_new = (1-d)/n + d * M @ r` — 这一行就是整个算法。

---

### 3.4 决策树 (信息增益)

| 项目 | 内容 |
|------|------|
| **一句话** | 每次选信息增益最大的特征切一刀，递归建树 |
| **口诀** | 算熵→算信息增益→选最优特征→递归 |
| **类型** | 有监督学习 / 分类 |
| **输入** | 特征矩阵 X, 标签 y, 特征名 |
| **输出** | 树结构（嵌套字典） |

**必背公式：**
```
熵:       H = -Σ (pᵢ * log₂(pᵢ))
信息增益: IG = H(父) - Σ 权重 * H(子)
```

**三个终止条件（必背）：**
1. 所有标签相同 → 返回该标签
2. 没有特征可分 → 返回多数类
3. 达到最大深度 → 返回多数类

**ID3 / C4.5 / CART 对比：**
- ID3: 信息增益（偏好多值特征）
- C4.5: 信息增益率（修正偏好）
- CART: 基尼系数（二叉树）

---

### 3.5 2D 卷积

| 项目 | 内容 |
|------|------|
| **一句话** | 卷积核在图像上滑动，逐元素乘后求和 |
| **口诀** | 双层循环→提取窗口→逐元素乘→求和 |
| **类型** | CNN 核心操作 |
| **输入** | 图像 (H×W), 卷积核 (kH×kW), 步长, 填充 |
| **输出** | 特征图 (out_H × out_W) |

**★ 必背公式（考试高频）：**
```
输出尺寸: out = (input - kernel + 2*padding) // stride + 1
卷积计算: output[i,j] = Σ(region * kernel)
```

**记忆口诀：** "输入减核加两倍填充，除步长加一"

---

### 3.6 最大池化

| 项目 | 内容 |
|------|------|
| **一句话** | 滑动窗口取最大值，实现下采样 |
| **口诀** | 双层循环→提取窗口→取最大值 |
| **类型** | CNN 下采样操作 |
| **输入** | 特征图, 池化窗口大小, 步长 |
| **输出** | 下采样后的特征图 |

**与卷积代码只差一行：**
```python
# 卷积: output[i,j] = np.sum(region * kernel)
# 池化: output[i,j] = np.max(region)
```

**关键区别：** 池化**没有可学习参数**，卷积核的值需要学习。

---

### 3.7 二分图判断

| 项目 | 内容 |
|------|------|
| **一句话** | BFS 交替染两色，相邻同色就不是二分图 |
| **口诀** | BFS→染色→检查冲突 |
| **类型** | 图论 / 图的性质判断 |
| **输入** | 邻接表 graph |
| **输出** | bool (是否为二分图) |

**必背代码模式：**
```python
color = {}
queue = deque([start])
color[start] = 0
while queue:
    node = queue.popleft()
    for nb in graph[node]:
        if nb not in color:
            color[nb] = 1 - color[node]  # ★ 交替染色
            queue.append(nb)
        elif color[nb] == color[node]:
            return False                  # ★ 冲突
```

**核心技巧：** `1 - color[node]` 实现 0↔1 交替。

**等价命题：** 图是二分图 ⟺ 图中无奇数环

---

### 3.8 Kruskal 最小生成树

| 项目 | 内容 |
|------|------|
| **一句话** | 边按权重排序，用并查集判环，贪心选边 |
| **口诀** | 排序边→并查集→逐边加入 |
| **类型** | 图论 / 最小生成树 |
| **输入** | 节点数 n, 边列表 [(w, u, v)] |
| **输出** | MST 边列表, 总权重 |

**并查集模板（必背）：**
```python
parent = list(range(n))  # 初始化
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # 路径压缩
    return parent[x]
def union(x, y):
    rx, ry = find(x), find(y)
    if rx == ry: return False  # 会成环
    parent[ry] = rx; return True
```

**MST 有 n-1 条边**（必记）

**Kruskal vs Prim：**
- Kruskal: 按边排序，适合稀疏图，O(E log E)
- Prim: 从点出发扩展，适合稠密图，O(V²)

---

### 3.9 梯度下降

| 项目 | 内容 |
|------|------|
| **一句话** | 沿损失函数梯度反方向更新参数 |
| **口诀** | 算预测→算误差→算梯度→更新参数 |
| **类型** | 优化算法 |
| **输入** | 特征矩阵 X, 目标 y, 学习率 lr |
| **输出** | 学习到的参数 w |

**★★★ 四行核心代码（必背）：**
```python
y_pred   = X @ w                      # 预测
error    = y_pred - y                  # 误差
gradient = (2/n) * X.T @ error         # 梯度
w        = w - lr * gradient           # 更新
```

**关键记忆：**
- 梯度公式中是 `X.T`（转置），不是 `X`
- 更新用**减号**（梯度指向上坡，要朝反方向走）
- 偏置处理：X 加一列全 1

**三种变体：** BGD(全部样本) / SGD(单个样本) / Mini-batch(一小批)

---

### 3.10 SMOTE 过采样

| 项目 | 内容 |
|------|------|
| **一句话** | 在少数类样本和其近邻之间随机插值生成新样本 |
| **口诀** | 找少数类→找K近邻→随机插值 |
| **类型** | 数据预处理 / 不平衡处理 |
| **输入** | 少数类样本, 需要生成的数量, K |
| **输出** | 合成的新样本 |

**★ 核心公式（必背）：**
```
x_new = x + λ * (x_neighbor - x),  λ ∈ [0, 1)
```
几何含义：在 x 和 x_neighbor 连线上随机取一点。

**关键细节：**
- 近邻索引从 `[1:k+1]` 开始，跳过自身 `[0]`
- 只在少数类样本之间操作
- SMOTE 只能在**训练集**上做，不能用于测试集

---

## 四、10 个算法的 NumPy 速查

| 操作 | NumPy 代码 | 出现在 |
|------|-----------|--------|
| 欧氏距离 | `np.linalg.norm(A - B, axis=1)` | KNN, K-Means, SMOTE |
| 升维广播 | `X[:, np.newaxis]` | K-Means |
| 排序取索引 | `np.argsort(arr)[:k]` | KNN, SMOTE |
| 布尔索引 | `X[labels == i]` | K-Means, 决策树 |
| 矩阵乘法 | `A @ B` 或 `X.T @ error` | PageRank, 梯度下降 |
| 加偏置列 | `np.column_stack([np.ones(n), x])` | 梯度下降 |
| 零填充 | `np.pad(img, p, constant_values=0)` | 卷积 |
| 计数统计 | `Counter(labels).most_common(1)` | KNN, 决策树 |
| 全近似比较 | `np.allclose(a, b)` | K-Means |
| 向量范数 | `np.linalg.norm(v)` | PageRank (收敛判断) |

---

## 五、考试高频考点速查

### 5.1 公式必背清单

| 算法 | 公式 | 说明 |
|------|------|------|
| KNN | `d = \|\|x-y\|\|₂` | 欧氏距离 |
| K-Means | `c = mean(簇内所有点)` | 簇心更新 |
| PageRank | `r = (1-d)/n + d*M@r` | 幂迭代 |
| 决策树 | `H = -Σ p·log₂(p)` | 信息熵 |
| 决策树 | `IG = H(父) - Σw·H(子)` | 信息增益 |
| 卷积/池化 | `out = (in-k+2p)//s + 1` | 输出尺寸 |
| 梯度下降 | `g = (2/n)·Xᵀ·(ŷ-y)` | MSE 梯度 |
| 梯度下降 | `w = w - lr·g` | 参数更新 |
| SMOTE | `x_new = x + λ·(xₙ-x)` | 插值公式 |

### 5.2 常考对比题

| 对比项 | 选项A | 选项B |
|--------|-------|-------|
| KNN vs K-Means | 有监督/分类, K=近邻数 | 无监督/聚类, K=簇数 |
| ID3 vs C4.5 | 信息增益，偏好多值特征 | 信息增益率，修正偏好 |
| Kruskal vs Prim | 按边排序，适合稀疏图 | 从点扩展，适合稠密图 |
| Max vs Avg Pooling | 保留最强特征 | 保留整体信息 |
| BGD vs SGD | 全部样本，稳定但慢 | 单个样本，快但不稳定 |
| 卷积 vs 池化 | 有可学习参数 | 无参数 |
| 过采样 vs 欠采样 | 增加少数类样本 | 减少多数类样本 |

### 5.3 复杂度速查

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| K-Means | O(n·k·d·T) | O(n·k) |
| KNN 预测 | O(n·d) | O(n·d) |
| PageRank | O(n²·T) | O(n²) |
| 决策树构建 | O(n·m·log n) | O(树大小) |
| 2D 卷积 | O(H'·W'·kH·kW) | O(H'·W') |
| Max Pooling | O(H'·W'·p²) | O(H'·W') |
| 二分图判断 | O(V+E) | O(V) |
| Kruskal | O(E log E) | O(V) |
| 梯度下降 | O(n·d·T) | O(d) |
| SMOTE | O(N_new·n·d) | O(N_new·d) |

---

## 六、考试答题技巧

1. **先写框架再填细节**：先把函数签名和 `for/while` 循环结构写出来，再填充内部逻辑
2. **变量命名要清晰**：`centroids`, `distances`, `labels` 等含义明确的变量名
3. **注释关键步骤**：在循环开头写一行注释说明这一步做什么
4. **NumPy 优先**：如果允许用库，NumPy 的向量化操作比 for 循环简洁得多
5. **边界条件**：注意输出矩阵的尺寸计算（卷积、池化）
6. **检查方向**：`axis=0` vs `axis=1`，`X` vs `X.T`，这些是最常见的错误来源
7. **默写检验法**：写完后用简单数据（如 2×2 矩阵）心算验证一遍

---

## 七、默写自测清单

完成学习后，用以下清单自测。每个算法限时 8 分钟，能在纸上写出核心函数即为过关。

- [ ] K-Means: 写出 `kmeans(X, k)` 函数（含初始化、分配、更新循环）
- [ ] KNN: 写出 `knn_predict(X_train, y_train, x_query, k)` 函数
- [ ] PageRank: 写出 `pagerank(adj, d=0.85)` 函数（含转移矩阵构建和幂迭代）
- [ ] 决策树: 写出 `entropy(labels)` 和 `information_gain(X_col, labels)` 函数
- [ ] 卷积: 写出 `conv2d(image, kernel, stride, padding)` 函数
- [ ] 池化: 写出 `max_pooling(image, pool_size, stride)` 函数
- [ ] 二分图: 写出 `is_bipartite(graph)` 函数（含 BFS 染色）
- [ ] Kruskal: 写出 `UnionFind` 类和 `kruskal(n, edges)` 函数
- [ ] 梯度下降: 写出 `gradient_descent(X, y, lr)` 函数（含四步循环）
- [ ] SMOTE: 写出 `smote(X_minority, n_new, k)` 函数

---

## 八、代码文件索引

| 文件 | 算法 | 核心概念 |
|------|------|----------|
| [`algorithms/01_kmeans.py`](algorithms/01_kmeans.py) | K-Means 聚类 | 簇心迭代 |
| [`algorithms/02_knn.py`](algorithms/02_knn.py) | K 近邻分类 | 欧氏距离 + 投票 |
| [`algorithms/03_pagerank.py`](algorithms/03_pagerank.py) | PageRank | 幂迭代 |
| [`algorithms/04_decision_tree.py`](algorithms/04_decision_tree.py) | 决策树 | 信息增益(ID3) |
| [`algorithms/05_convolution.py`](algorithms/05_convolution.py) | 2D 卷积 | 滑动窗口 |
| [`algorithms/06_max_pooling.py`](algorithms/06_max_pooling.py) | 最大池化 | 滑动窗口 |
| [`algorithms/07_bipartite_check.py`](algorithms/07_bipartite_check.py) | 二分图判断 | BFS 染色 |
| [`algorithms/08_kruskal.py`](algorithms/08_kruskal.py) | 最小生成树 | Kruskal + 并查集 |
| [`algorithms/09_gradient_descent.py`](algorithms/09_gradient_descent.py) | 梯度下降 | 线性回归 |
| [`algorithms/10_smote.py`](algorithms/10_smote.py) | SMOTE 过采样 | K近邻插值 |

每个文件都包含：
- 详细的算法背景与原理注释
- 流程图 (ASCII art)
- 记忆要点与考试必背内容
- 易错点提醒
- 复杂度分析
- 核心实现函数
- 可直接运行的示例 + 理解辅助输出
