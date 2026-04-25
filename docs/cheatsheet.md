# 考试速查表 (Cheatsheet)

> 考前 30 分钟快速过一遍。只保留最核心的公式、代码和对比。

---

## 一、10 个算法一句话 + 口诀

| # | 算法 | 一句话 | 口诀 |
|---|------|--------|------|
| 01 | K-Means | 随机选簇心，反复分配+更新直到不动 | 初始化→分配→更新→重复 |
| 02 | KNN | 算距离找最近K个投票 | 算距离→排序→投票 |
| 03 | PageRank | 转移矩阵反复乘排名向量 | 转移矩阵→幂迭代→归一化 |
| 04 | 决策树 | 选信息增益最大的特征递归分裂 | 算熵→算IG→选特征→递归 |
| 05 | 卷积 | 核在图上滑动做加权求和 | 双层循环→提窗口→乘→求和 |
| 06 | 池化 | 窗口滑动取最大值 | 双层循环→提窗口→取max |
| 07 | 二分图 | BFS交替染色查冲突 | BFS→染色→查冲突 |
| 08 | Kruskal | 边排序+并查集贪心选边 | 排序→并查集→逐边加 |
| 09 | 梯度下降 | 沿梯度反方向更新参数 | 预测→误差→梯度→更新 |
| 10 | SMOTE | 少数类近邻间随机插值 | 选样本→找近邻→插值 |

---

## 二、必背公式

### 距离与度量
```
欧氏距离: d(x,y) = ||x - y||₂ = sqrt(Σ(xᵢ - yᵢ)²)
```

### K-Means
```
簇心更新: c_new = (1/|S|) * Σ x,   x ∈ S
```

### PageRank ★★★
```
r_new = (1-d)/n + d * M @ r     d=0.85, M=转移矩阵
```

### 决策树
```
信息熵: H(S) = -Σ pᵢ * log₂(pᵢ)
信息增益: IG(S,A) = H(S) - Σ (|Sᵥ|/|S|) * H(Sᵥ)
```

### 卷积/池化 ★★★
```
输出尺寸: out = (input - kernel + 2*padding) // stride + 1
记忆法: "输入减核加两倍填充，除步长加一"
```

### 梯度下降 (线性回归) ★★★
```
预测:   ŷ = X @ w
梯度:   g = (2/n) * Xᵀ @ (ŷ - y)
更新:   w = w - lr * g
```

### SMOTE
```
x_new = x + λ * (x_neighbor - x),   λ ~ U(0,1)
```

### 并查集
```
find(x): if parent[x]!=x: parent[x]=find(parent[x]); return parent[x]
union(x,y): rx,ry=find(x),find(y); if rx==ry: False; parent[ry]=rx
```

---

## 三、常考对比

| 对比项 | A | B |
|--------|---|---|
| **KNN vs K-Means** | 有监督/分类, K=近邻数 | 无监督/聚类, K=簇数 |
| **ID3 vs C4.5 vs CART** | 信息增益 / 信息增益率 / 基尼系数 | 偏好多值 / 修正偏好 / 二叉树 |
| **Kruskal vs Prim** | 按边排序, O(ElogE), 稀疏图 | 从点扩展, O(V²), 稠密图 |
| **Max vs Avg Pooling** | 保留最强特征, 更常用 | 保留整体信息 |
| **BGD vs SGD vs Mini-batch** | 全部样本/稳定慢 | 1个/快不稳 | 一批/平衡 |
| **卷积 vs 池化** | 有可学习参数(卷积核) | 无参数 |
| **过采样 vs 欠采样** | 增加少数类(SMOTE) | 减少多数类(丢信息) |

---

## 四、复杂度速查

| 算法 | 时间 | 空间 |
|------|------|------|
| K-Means | O(n·k·d·T) | O(n·k) |
| KNN 预测 | O(n·d) | O(n·d) |
| PageRank | O(n²·T) | O(n²) |
| 决策树构建 | O(n·m·logn) | O(树) |
| 2D 卷积 | O(H'W'·kHkW) | O(H'W') |
| Max Pooling | O(H'W'·p²) | O(H'W') |
| 二分图 | O(V+E) | O(V) |
| Kruskal | O(ElogE) | O(V) |
| 梯度下降 | O(n·d·T) | O(d) |
| SMOTE | O(N·n·d) | O(N·d) |

---

## 五、NumPy 关键操作速查

| 操作 | 代码 | 用于 |
|------|------|------|
| 欧氏距离 | `np.linalg.norm(A-B, axis=1)` | KNN, K-Means, SMOTE |
| 升维广播 | `X[:, np.newaxis]` | K-Means 批量距离 |
| 排序取索引 | `np.argsort(arr)[:k]` | KNN, SMOTE |
| 布尔索引 | `X[labels == i]` | K-Means, 决策树 |
| 矩阵乘法 | `X.T @ error` | PageRank, 梯度下降 |
| 加偏置列 | `np.column_stack([np.ones(n), x])` | 梯度下降 |
| 零填充 | `np.pad(img, p, constant_values=0)` | 卷积 |
| 计数投票 | `Counter(labels).most_common(1)` | KNN, 决策树 |
| 近似比较 | `np.allclose(a, b)` | K-Means 收敛 |

---

## 六、易错点速查

| 算法 | 常见错误 |
|------|----------|
| K-Means | 空簇未处理；忘记 `np.newaxis` 做广播 |
| KNN | `axis` 方向搞反；忘做特征标准化 |
| PageRank | 转移矩阵忘转置；悬挂节点除零 |
| 决策树 | 用 `np.log` 而非 `np.log2`；分裂后忘删已用特征列 |
| 卷积 | 输出尺寸公式记错；`region*kernel` 写成 `@` |
| 池化 | stride ≠ pool_size 时计算错 |
| 二分图 | 忘处理多个连通分量 |
| Kruskal | 路径压缩写成 `parent[parent[x]]`；终止条件 n 而非 n-1 |
| 梯度下降 | 梯度公式忘 `X.T`；更新时加号写成减号 |
| SMOTE | 近邻索引没跳过自身 `[0]` |
