# AI 算法一日速通指南

> 目标：在一天内掌握 10 个高频考试算法的 Python 核心实现，能默写关键代码，理解原理。

---

## 一、学习总策略

```
理解原理 → 看代码 → 手写 → 运行验证 → 总结口诀
```

每个算法只需掌握：
1. **一句话原理**（能说清楚在做什么）
2. **核心数据结构**（用什么存储中间状态）
3. **关键循环/递推逻辑**（5~15 行代码核心）
4. **终止条件**（什么时候停止）

---

## 二、一日学习时间表（8 小时）

| 时间段 | 内容 | 预计时长 |
|--------|------|---------|
| 08:00–09:00 | 机器学习基础：K-Means + KNN | 60 min |
| 09:00–10:00 | 优化与采样：梯度下降 + SMOTE | 60 min |
| 10:00–10:15 | 休息 | 15 min |
| 10:15–11:15 | 图算法：PageRank + 二分图判断 | 60 min |
| 11:15–12:00 | 图算法：Kruskal 最小生成树 | 45 min |
| 12:00–13:00 | 午休 | 60 min |
| 13:00–14:00 | 深度学习基础：决策树信息增益 | 60 min |
| 14:00–15:00 | 深度学习基础：卷积 + Max Pooling | 60 min |
| 15:00–15:15 | 休息 | 15 min |
| 15:15–17:00 | 全部算法默写练习 + 查漏补缺 | 105 min |
| 17:00–18:00 | 模拟考试：不看代码独立复现所有算法 | 60 min |

---

## 三、算法详解与记忆口诀

### 1. K-Means（聚类）

**一句话原理**：随机初始化 K 个中心点，反复将每个样本分配给最近的中心，再更新中心为所属簇的均值，直到中心不再变化。

**记忆口诀**：`随机选中心 → 分配最近簇 → 更新均值 → 重复直到收敛`

**关键代码骨架**：
```python
for _ in range(max_iter):
    # 1. 分配：每个点找最近中心
    labels = [argmin(dist(x, center)) for x in X]
    # 2. 更新：每个簇取均值
    new_centers = [mean(X[labels==k]) for k in range(K)]
    if 收敛: break
    centers = new_centers
```

**考试关键点**：
- 距离用欧氏距离（L2）
- 初始化可随机选 K 个样本
- 终止：中心点不变 或 达到最大迭代次数

---

### 2. KNN（分类/回归）

**一句话原理**：预测新样本时，找训练集中欧氏距离最近的 K 个邻居，投票（分类）或取均值（回归）。

**记忆口诀**：`算距离 → 排序取 K 近 → 投票/均值`

**关键代码骨架**：
```python
def predict(x):
    dists = [euclidean(x, xi) for xi in X_train]
    k_idx = argsort(dists)[:K]
    return majority_vote(y_train[k_idx])
```

**考试关键点**：
- 无训练过程，是 lazy learning
- K 是超参数，K 越大决策边界越平滑
- 欧氏距离：`sqrt(sum((a-b)**2))`

---

### 3. PageRank（网页排名）

**一句话原理**：网页排名 = 所有指向它的网页排名之和（按出链数均分），加上随机跳转概率，用幂迭代反复更新直到收敛。

**记忆口诀**：`PR = d * 入链贡献之和 + (1-d)/N`

**关键代码骨架**：
```python
for _ in range(max_iter):
    new_rank = {}
    for node in graph:
        new_rank[node] = (1-d)/N + d * sum(
            rank[src] / out_degree[src]
            for src in in_links[node]
        )
    if 收敛: break
    rank = new_rank
```

**考试关键点**：
- 阻尼系数 d 通常取 0.85
- 初始 PR 值均为 1/N
- 终止：各节点 PR 变化 < 阈值

---

### 4. 决策树——信息增益

**一句话原理**：选择使信息增益最大的特征作为分裂点。信息增益 = 父节点熵 - 子节点加权熵之和。

**记忆口诀**：`熵(父) - Σ(子节点比例 × 熵(子))` → 取最大

**关键代码骨架**：
```python
def entropy(y):
    p = counts(y) / len(y)
    return -sum(p * log2(p))

def info_gain(X_col, y):
    parent_H = entropy(y)
    for val in unique(X_col):
        child_H += (len(子集)/len(y)) * entropy(y[X_col==val])
    return parent_H - child_H
```

**考试关键点**：
- 熵越小越纯
- 选信息增益最大的特征分裂
- ID3 用信息增益，C4.5 用增益率，CART 用基尼系数

---

### 5. 卷积（滑动窗口）

**一句话原理**：用小尺寸的卷积核在输入矩阵上逐步滑动，每次对重叠区域做逐元素乘法并求和，得到输出特征图。

**记忆口诀**：`输出[i][j] = sum(输入[i:i+k, j:j+k] * 卷积核)`

**关键代码骨架**：
```python
for i in range(output_h):
    for j in range(output_w):
        output[i][j] = sum(
            input[i*s:i*s+kh, j*s:j*s+kw] * kernel
        )
```

**考试关键点**：
- 步长 stride 控制滑动步幅
- padding 控制边界补零
- 输出尺寸：`(N - K + 2P) / S + 1`

---

### 6. Max Pooling（最大池化）

**一句话原理**：在输入矩阵上用固定大小的窗口滑动，每次取窗口内的最大值，起到降采样和特征增强的作用。

**记忆口诀**：`输出[i][j] = max(输入[i*s:i*s+k, j*s:j*s+k])`

**关键代码骨架**：
```python
for i in range(output_h):
    for j in range(output_w):
        output[i][j] = max(
            input[i*s:i*s+pool_h, j*s:j*s+pool_w]
        )
```

**考试关键点**：
- 与卷积的区别：无可学习参数，只取最大值
- Average Pooling 取均值
- 输出尺寸与卷积相同公式

---

### 7. 二分图判断（BFS 染色）

**一句话原理**：用 BFS 从任意节点出发，相邻节点染不同颜色（0/1），若发现相邻节点同色则不是二分图。

**记忆口诀**：`BFS 染色 → 相邻不同色 → 冲突即非二分图`

**关键代码骨架**：
```python
color = {start: 0}
queue = [start]
while queue:
    node = queue.pop()
    for neighbor in graph[node]:
        if neighbor not in color:
            color[neighbor] = 1 - color[node]
            queue.append(neighbor)
        elif color[neighbor] == color[node]:
            return False  # 不是二分图
return True
```

**考试关键点**：
- 需处理非连通图（对每个连通分量都跑一遍）
- 颜色翻转用 `1 - color[node]`

---

### 8. 最小生成树——Kruskal

**一句话原理**：按边权从小到大排序，依次添加不形成环的边（用并查集判断），直到选了 N-1 条边。

**记忆口诀**：`排边权 → 贪心加边 → 并查集判环 → N-1 条边结束`

**关键代码骨架**：
```python
edges.sort(key=lambda e: e[2])  # 按权重排序
uf = UnionFind(n)
for u, v, w in edges:
    if uf.find(u) != uf.find(v):  # 不成环
        uf.union(u, v)
        mst.append((u, v, w))
        if len(mst) == n-1: break
```

**考试关键点**：
- 并查集 find 要路径压缩
- union 要按秩合并
- Prim 是另一种 MST 算法（适合稠密图）

---

### 9. 梯度下降（线性回归）

**一句话原理**：沿着损失函数对参数的梯度的反方向更新参数，重复直到损失收敛。

**记忆口诀**：`预测 → 计算误差 → 求梯度 → 反向更新参数`

**关键代码骨架**：
```python
for _ in range(epochs):
    y_pred = X @ w + b          # 前向传播
    error = y_pred - y_true     # 误差
    grad_w = X.T @ error / n    # 梯度
    grad_b = error.mean()
    w -= lr * grad_w            # 参数更新
    b -= lr * grad_b
```

**考试关键点**：
- 损失函数：MSE = mean((pred - true)²)
- 学习率 lr 太大会震荡，太小收敛慢
- 批量梯度下降 vs 随机梯度下降（SGD）

---

### 10. SMOTE（过采样）

**一句话原理**：对少数类样本，找其 K 个最近邻，在该样本与随机选择的邻居之间的连线上随机插值，生成新的合成样本。

**记忆口诀**：`少数类 → K近邻 → 随机插值生成新样本`

**关键代码骨架**：
```python
for x in minority_samples:
    neighbors = knn(x, k=K)
    xn = random.choice(neighbors)
    # 在 x 到 xn 之间随机插值
    synthetic = x + random.uniform(0, 1) * (xn - x)
    new_samples.append(synthetic)
```

**考试关键点**：
- 解决类别不平衡问题
- 插值公式：`new = x + λ * (xn - x)`，λ ∈ [0, 1]
- 只对少数类做过采样

---

## 四、考前速查：算法对比表

| 算法 | 类别 | 核心操作 | 关键参数 |
|------|------|---------|---------|
| K-Means | 无监督聚类 | 欧氏距离 + 均值更新 | K（簇数）|
| KNN | 监督分类 | 欧氏距离 + 投票 | K（邻居数）|
| PageRank | 图排名 | 幂迭代 | d（阻尼系数=0.85）|
| 决策树 | 监督分类 | 信息增益/熵 | 深度限制 |
| 卷积 | 特征提取 | 滑动窗口乘加 | 核大小、步长 |
| Max Pooling | 降采样 | 滑动窗口取最大 | 窗口大小、步长 |
| 二分图 | 图判断 | BFS 染色 | 无 |
| Kruskal | 最小生成树 | 贪心 + 并查集 | 无 |
| 梯度下降 | 优化 | 反向传播更新参数 | 学习率 lr |
| SMOTE | 数据增强 | K近邻 + 插值 | K（邻居数）|

---

## 五、代码文件说明

```
/workspace
├── STUDY_GUIDE.md         # 本学习指南
├── machine_learning.py    # K-Means、KNN、梯度下降、SMOTE
├── graph_algorithms.py    # PageRank、二分图判断、Kruskal
├── dl_basics.py           # 决策树(信息增益)、卷积、Max Pooling
└── run_all.py             # 一键运行所有算法示例，验证输出
```

---

## 六、默写练习方法

1. **看一遍代码**（2 分钟）：重点看循环结构和变量名
2. **合上代码手写**（5 分钟）：只写核心逻辑，不抠细节
3. **对比查漏**（2 分钟）：找出遗漏的步骤
4. **再手写一遍**（3 分钟）：这次不看任何提示

每个算法重复以上步骤 2–3 次，一般即可记牢。

---

## 七、高频考点提示

- **K-Means vs KNN**：一个是无监督聚类（K 是簇数），一个是有监督分类（K 是邻居数）
- **信息增益的熵公式**：`H = -Σ p_i * log2(p_i)`，注意 `0*log(0)=0`
- **PageRank 的阻尼系数**：d=0.85，代表 85% 概率沿链接跳转，15% 随机跳转
- **Kruskal 必须掌握并查集**：find（路径压缩）和 union（按秩合并）
- **卷积输出尺寸公式**：`(N - K + 2P) / S + 1`，务必记住
- **SMOTE 插值**：新样本在两点连线上，不是随机点
