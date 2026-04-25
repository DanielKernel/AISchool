# AI 算法一天速成学习指南

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

> 总时长约 **8 小时**，包含学习 + 练习 + 模拟考试。每个阶段结束后务必做一次默写练习。

### 第一阶段：迭代收敛组（2 小时）

| 时间 | 内容 | 资源 |
|------|------|------|
| 30 min | **K-Means**：阅读代码 + 运行示例 + 理解距离广播 | `algorithms/01_kmeans.py` |
| 30 min | **PageRank**：阅读代码 + 理解转移矩阵构建 | `algorithms/03_pagerank.py` |
| 30 min | **梯度下降**：阅读代码 + 理解梯度公式推导 | `algorithms/09_gradient_descent.py` |
| 30 min | **默写练习**：合上屏幕，在纸上默写 3 个核心循环 | 纸笔 |

**记忆锚点**：三个算法都是 `while not converged: update()` 模式。区别只在「更新什么」：
- K-Means → 更新**簇心**（均值）
- PageRank → 更新**排名向量**（矩阵乘法）
- 梯度下降 → 更新**模型参数**（减去梯度）

### 第二阶段：分类/决策组（1.5 小时）

| 时间 | 内容 | 资源 |
|------|------|------|
| 30 min | **KNN**：阅读代码 + 手工追踪距离计算 | `algorithms/02_knn.py` |
| 30 min | **决策树**：阅读代码 + 手算一次信息增益 | `algorithms/04_decision_tree.py` |
| 30 min | **默写练习** + 做填空题 | `exercises/01_fill_in_the_blank.py` |

**记忆锚点**：KNN 是「找最近的 K 个邻居投票」，决策树是「找最好的特征切一刀，递归」。

### 第三阶段：矩阵/窗口运算组（1 小时）

| 时间 | 内容 | 资源 |
|------|------|------|
| 20 min | **卷积**：阅读代码 + 手算 output[0,0] | `algorithms/05_convolution.py` |
| 20 min | **池化**：阅读代码 + 对比卷积的唯一差异 | `algorithms/06_max_pooling.py` |
| 20 min | **默写练习** + 做填空题 | `exercises/01_fill_in_the_blank.py` |

**记忆锚点**：两个算法代码框架**几乎一模一样**，只有窗口内的操作不同。

### 第四阶段：图算法组（1.5 小时）

| 时间 | 内容 | 资源 |
|------|------|------|
| 35 min | **二分图判断**：阅读代码 + 手画 BFS 染色过程 | `algorithms/07_bipartite_check.py` |
| 35 min | **Kruskal MST**：阅读代码 + 手动模拟并查集 | `algorithms/08_kruskal.py` |
| 20 min | **默写练习** + 做填空题 | `exercises/01_fill_in_the_blank.py` |

**记忆锚点**：二分图用 BFS 队列 + 颜色字典，Kruskal 用排序 + 并查集。

### 第五阶段：数据增强 + 填空练习（1 小时）

| 时间 | 内容 | 资源 |
|------|------|------|
| 20 min | **SMOTE**：阅读代码 + 理解插值公式 | `algorithms/10_smote.py` |
| 40 min | **全部填空练习**：完成所有练习题并对答案 | `exercises/01_fill_in_the_blank.py` |

### 第六阶段：模拟考试 + 查漏补缺（1.5 小时）

| 时间 | 内容 | 资源 |
|------|------|------|
| 60 min | **模拟考试**：限时完成 10 道题 | `exercises/02_mock_exam.py` |
| 30 min | **对答案 + 补强薄弱点** | 运行 `python3 exercises/02_mock_exam.py` |

---

## 三、每个算法记忆卡片

> 每张卡片包含：一句话总结 / 口诀 / 必背代码骨架 / 关键技巧。
> 更详细的背景、流程图、易错点请阅读 `algorithms/` 目录下对应的源代码文件。

### 3.1 K-Means 聚类

```
口诀：初始化 → 分配 → 更新 → 重复
类型：无监督 / 聚类
```

```python
centroids = X[random_indices]         # 初始化
for _ in range(max_iters):
    distances = norm(X - centroids)   # 算距离
    labels = argmin(distances)        # 分配
    centroids = mean(X[labels==i])    # 更新
    if 不变了: break                  # 收敛
```

### 3.2 KNN 分类

```
口诀：算距离 → 排序 → 投票
类型：有监督 / 分类（懒惰学习）
```

```python
distances = norm(X_train - x_query)   # 算距离
k_idx = argsort(distances)[:k]        # 排序取前K
labels = y_train[k_idx]              # 取标签
return Counter(labels).most_common(1) # 投票
```

### 3.3 PageRank

```
口诀：构建转移矩阵 → 幂迭代 → 归一化
类型：图算法 / 链接分析
★ 核心就一行：r_new = (1-d)/n + d * M @ r
```

```python
M = (adj / out_degree).T             # 转移矩阵
r = ones(n) / n                      # 初始排名
for _ in range(max_iters):
    r_new = (1-d)/n + d * M @ r      # 幂迭代
    if 收敛: break
```

### 3.4 决策树 (信息增益)

```
口诀：算熵 → 算信息增益 → 选最优特征 → 递归
必背公式：H = -Σ p·log₂(p)
         IG = H(父) - Σ w·H(子)
三个终止条件：标签纯净 / 无特征 / 达最大深度
```

### 3.5 & 3.6 卷积与池化

```
口诀：双层循环 → 提取窗口 → (卷积)逐元素乘求和 / (池化)取最大值
★ 必背公式：out = (input - kernel + 2*padding) // stride + 1
```

```python
for i in range(out_H):
    for j in range(out_W):
        region = image[i*s:i*s+kH, j*s:j*s+kW]
        output[i,j] = np.sum(region * kernel)  # 卷积
        # 或者
        output[i,j] = np.max(region)           # 池化
```

### 3.7 二分图判断

```
口诀：BFS → 染色 → 检查冲突
核心技巧：color[nb] = 1 - color[node] 实现交替染色
等价命题：无奇数环 ⟺ 二分图
```

### 3.8 Kruskal 最小生成树

```
口诀：排序边 → 并查集 → 逐边加入（不成环就加）
MST 有 n-1 条边
并查集 find 必须路径压缩
```

### 3.9 梯度下降

```
口诀：算预测 → 算误差 → 算梯度 → 更新参数
```

```python
y_pred   = X @ w                      # 预测
error    = y_pred - y                  # 误差
gradient = (2/n) * X.T @ error         # 梯度（注意 X.T）
w        = w - lr * gradient           # 更新（注意减号）
```

### 3.10 SMOTE 过采样

```
口诀：找少数类 → 找K近邻 → 随机插值
★ 核心公式：x_new = x + λ * (x_neighbor - x),  λ ∈ [0,1)
注意：近邻从 [1:k+1] 取，跳过自身
```

---

## 四、考试速查表

> 详细速查表请参见 [`docs/cheatsheet.md`](cheatsheet.md)

---

## 五、常见考试题型与应对

### 题型 1：代码填空
给出算法框架，要求填写关键行。练习资源：`exercises/01_fill_in_the_blank.py`

### 题型 2：手写完整函数
给出函数签名，要求写完整实现。每个算法限时 8 分钟。

### 题型 3：概念辨析
如 "KNN 与 K-Means 的区别"、"ID3 与 C4.5 的区别" 等。

### 题型 4：手算
给定小数据，要求手算一步或多步。如：
- 给 3 个点和 2 个簇心，手算一次 K-Means 迭代
- 给 4x4 矩阵和 2x2 核，手算卷积/池化结果
- 给 5 个有标签样本，计算某特征的信息增益

### 题型 5：复杂度分析
要求给出算法的时间/空间复杂度。

---

## 六、默写自测清单

完成学习后，用以下清单自测。每个限时 8 分钟，能写出核心函数即为过关。

- [ ] K-Means: 写出 `kmeans(X, k)` 函数
- [ ] KNN: 写出 `knn_predict(X_train, y_train, x_query, k)` 函数
- [ ] PageRank: 写出 `pagerank(adj, d=0.85)` 函数
- [ ] 决策树: 写出 `entropy(labels)` 和 `information_gain(X_col, labels)` 函数
- [ ] 卷积: 写出 `conv2d(image, kernel, stride, padding)` 函数
- [ ] 池化: 写出 `max_pooling(image, pool_size, stride)` 函数
- [ ] 二分图: 写出 `is_bipartite(graph)` 函数
- [ ] Kruskal: 写出 `UnionFind` 类和 `kruskal(n, edges)` 函数
- [ ] 梯度下降: 写出 `gradient_descent(X, y, lr)` 函数
- [ ] SMOTE: 写出 `smote(X_minority, n_new, k)` 函数
