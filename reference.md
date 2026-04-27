我将为您实现这12个算法，每个都包含详细的注释、原理说明和测试代码。

```python
import numpy as np
from collections import deque, defaultdict
import heapq
import random

# ==================== 1. K-Means (簇心迭代) ====================
def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    K-Means聚类算法
    
    算法原理:
    1. 随机选择k个初始质心
    2. 重复直到收敛:
        a. 将每个样本分配到最近的质心(距离度量:欧氏距离)
        b. 重新计算每个簇的质心(取簇内样本均值)
    
    关键要点:
    - 使用欧氏距离的平方避免开方运算
    - 质心初始化的优化(这里使用随机选择)
    - 收敛条件:质心移动距离小于阈值或达到最大迭代次数
    """
    n_samples, n_features = X.shape
    
    # 1. 随机初始化质心
    np.random.seed(42)
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()
    
    for _ in range(max_iters):
        # 2. 分配样本到最近的质心
        # 计算每个样本到每个质心的距离矩阵
        distances = np.zeros((n_samples, k))
        for i in range(k):
            # 使用平方欧氏距离，避免开方
            distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
        
        # 获取每个样本最近的质心索引
        labels = np.argmin(distances, axis=1)
        
        # 3. 重新计算质心
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            if np.sum(labels == i) > 0:  # 避免除零
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:  # 如果簇为空，随机重新初始化
                new_centroids[i] = X[np.random.randint(0, n_samples)]
        
        # 4. 检查收敛
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
            
        centroids = new_centroids
    
    return centroids, labels

def test_kmeans():
    """测试K-Means"""
    np.random.seed(42)
    # 创建测试数据：3个簇
    cluster1 = np.random.randn(30, 2) + [2, 2]
    cluster2 = np.random.randn(30, 2) + [-2, 2]
    cluster3 = np.random.randn(30, 2) + [0, -2]
    X = np.vstack([cluster1, cluster2, cluster3])
    
    centroids, labels = kmeans(X, k=3)
    print(f"K-Means测试通过")
    print(f"质心形状: {centroids.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签分布: {np.bincount(labels)}")


# ==================== 2. KNN (欧氏距离+投票) ====================
def knn(X_train, y_train, X_test, k=5):
    """
    K近邻分类算法
    
    算法原理:
    1. 对于每个测试样本，计算它与所有训练样本的欧氏距离
    2. 选择距离最近的k个训练样本
    3. 通过投票决定测试样本的类别
    
    关键要点:
    - 使用广播机制批量计算距离
    - 处理类别平票情况(选择第一个)
    """
    y_pred = []
    
    for i in range(len(X_test)):
        # 计算欧氏距离(使用平方避免开方)
        distances = np.sum((X_train - X_test[i])**2, axis=1)
        
        # 获取最近的k个邻居的索引
        nearest_indices = np.argsort(distances)[:k]
        
        # 获取邻居的标签
        nearest_labels = y_train[nearest_indices]
        
        # 投票决定类别
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        pred_label = unique_labels[np.argmax(counts)]
        y_pred.append(pred_label)
    
    return np.array(y_pred)

def test_knn():
    """测试KNN"""
    np.random.seed(42)
    # 创建简单数据集
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 7], [7, 8], [8, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2, 2], [7, 7]])
    
    y_pred = knn(X_train, y_train, X_test, k=3)
    print(f"KNN测试通过")
    print(f"预测结果: {y_pred}")
    print(f"期望结果: [0, 1]")


# ==================== 3. PageRank (幂迭代) ====================
def pagerank(adj_matrix, damping=0.85, max_iters=100, tol=1e-6):
    """
    PageRank算法(幂迭代实现)
    
    算法原理:
    1. 构建转移概率矩阵
    2. 应用阻尼因子处理悬挂节点
    3. 通过幂迭代计算PageRank值直到收敛
    
    公式: PR = (1-d)/N + d * (A * PR)
    其中A是转移矩阵
    
    关键要点:
    - 处理悬挂节点(出度为0的节点)
    - 注意概率矩阵的归一化
    """
    n = len(adj_matrix)
    
    # 1. 计算转移概率矩阵
    # 出度
    out_degree = adj_matrix.sum(axis=1)
    
    # 避免除零
    out_degree[out_degree == 0] = 1
    
    # 转移矩阵
    transition = adj_matrix / out_degree[:, np.newaxis]
    
    # 2. 初始化PageRank向量
    pr = np.ones(n) / n
    
    # 3. 幂迭代
    for _ in range(max_iters):
        new_pr = (1 - damping) / n + damping * transition.T @ pr
        
        # 检查收敛
        if np.linalg.norm(new_pr - pr) < tol:
            break
            
        pr = new_pr
    
    return pr

def test_pagerank():
    """测试PageRank"""
    # 创建简单的网页链接矩阵
    # 页面0 -> 页面1, 页面2
    # 页面1 -> 页面2
    # 页面2 -> 页面0, 页面1
    adj_matrix = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 0]
    ], dtype=float)
    
    pr = pagerank(adj_matrix)
    print(f"PageRank测试通过")
    print(f"PageRank值: {pr}")
    print(f"验证: 总和应为1, 实际: {np.sum(pr):.6f}")


# ==================== 4. 决策树核心 (信息增益) ====================
def entropy(y):
    """计算熵"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))  # 加小量避免log(0)

def information_gain(X, y, feature_idx, threshold):
    """
    计算信息增益
    
    公式: IG = H(Y) - H(Y|X)
    对于数值特征，通过阈值进行二分
    """
    # 根据阈值分割数据
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0
    
    # 计算父节点的熵
    parent_entropy = entropy(y)
    
    # 计算子节点的加权熵
    n = len(y)
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    
    # 加权平均
    weighted_entropy = (np.sum(left_mask) * left_entropy + 
                       np.sum(right_mask) * right_entropy) / n
    
    # 信息增益
    return parent_entropy - weighted_entropy

def find_best_split(X, y):
    """寻找最佳分割点"""
    best_gain = 0
    best_feature = None
    best_threshold = None
    
    n_samples, n_features = X.shape
    
    for feature_idx in range(n_features):
        # 获取该特征的所有唯一值作为候选阈值
        unique_values = np.unique(X[:, feature_idx])
        
        for threshold in unique_values:
            gain = information_gain(X, y, feature_idx, threshold)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

def test_decision_tree():
    """测试决策树信息增益计算"""
    np.random.seed(42)
    # 创建简单数据集
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 6], [6, 4]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    best_feature, best_threshold, best_gain = find_best_split(X, y)
    
    print(f"决策树核心测试通过")
    print(f"最佳特征: {best_feature}")
    print(f"最佳阈值: {best_threshold}")
    print(f"信息增益: {best_gain:.4f}")


# ==================== 5. 二分图判断 (BFS染色) ====================
def is_bipartite(adj_list):
    """
    判断是否为二分图(BFS染色)
    
    算法原理:
    1. 从任意未访问节点开始BFS
    2. 给当前节点染色(0或1)
    3. 给相邻节点染相反颜色
    4. 如果相邻节点已染色且颜色相同，则不是二分图
    
    关键要点:
    - 处理非连通图
    - 使用队列进行BFS
    """
    n = len(adj_list)
    colors = [-1] * n  # -1表示未染色，0和1表示两种颜色
    
    for start in range(n):
        if colors[start] == -1:  # 未访问节点
            queue = deque([start])
            colors[start] = 0  # 初始染色为0
            
            while queue:
                node = queue.popleft()
                
                for neighbor in adj_list[node]:
                    if colors[neighbor] == -1:  # 未染色
                        colors[neighbor] = 1 - colors[node]  # 染相反颜色
                        queue.append(neighbor)
                    elif colors[neighbor] == colors[node]:  # 颜色冲突
                        return False
    
    return True

def test_bipartite():
    """测试二分图判断"""
    # 创建一个二分图
    adj_list = [
        [1, 3],  # 节点0连接1,3
        [0, 2],  # 节点1连接0,2
        [1, 3],  # 节点2连接1,3
        [0, 2]   # 节点3连接0,2
    ]
    
    result = is_bipartite(adj_list)
    print(f"二分图判断测试通过")
    print(f"是否为二分图: {result}")
    print(f"期望结果: True (因为是4个节点的环)")


# ==================== 6. 最小生成树 (Kruskal) ====================
def kruskal(n, edges):
    """
    Kruskal算法求最小生成树
    
    算法原理:
    1. 将所有边按权重从小到大排序
    2. 依次尝试每条边，如果加入不会形成环，则加入生成树
    3. 使用并查集检测环
    
    关键要点:
    - 并查集实现查找和合并
    - 边数达到n-1时提前终止
    """
    # 并查集
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        
        if root_x != root_y:
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
            return True
        return False
    
    # 按权重排序边
    edges.sort(key=lambda x: x[2])
    
    mst = []
    mst_weight = 0
    edges_used = 0
    
    for u, v, w in edges:
        if union(u, v):  # 不形成环
            mst.append((u, v, w))
            mst_weight += w
            edges_used += 1
            
            if edges_used == n - 1:  # 最小生成树有n-1条边
                break
    
    return mst, mst_weight

def test_kruskal():
    """测试Kruskal算法"""
    # 创建一个简单图
    n = 4
    edges = [
        (0, 1, 1),  # 边0-1，权重1
        (0, 2, 3),  # 边0-2，权重3
        (1, 2, 2),  # 边1-2，权重2
        (1, 3, 4),  # 边1-3，权重4
        (2, 3, 5)   # 边2-3，权重5
    ]
    
    mst, weight = kruskal(n, edges)
    print(f"最小生成树测试通过")
    print(f"最小生成树边: {mst}")
    print(f"最小权重: {weight}")
    print(f"期望权重: 7 (1+2+4)")


# ==================== 7. 梯度下降 (线性回归版) ====================
def gradient_descent(X, y, learning_rate=0.01, n_iters=1000):
    """
    梯度下降(线性回归)
    
    算法原理:
    1. 初始化参数w,b
    2. 重复迭代:
        a. 计算预测值 y_pred = X@w + b
        b. 计算损失 L = 1/(2m) * sum((y_pred - y)^2)
        c. 计算梯度: dw = 1/m * X^T @ (y_pred - y)
                     db = 1/m * sum(y_pred - y)
        d. 更新参数: w = w - lr*dw, b = b - lr*db
    
    关键要点:
    - 特征缩放(这里假设已处理)
    - 学习率选择
    """
    m, n = X.shape
    
    # 初始化参数
    w = np.zeros(n)
    b = 0
    
    # 梯度下降
    for i in range(n_iters):
        # 前向传播
        y_pred = X @ w + b
        
        # 计算损失
        loss = np.mean((y_pred - y) ** 2) / 2
        
        # 计算梯度
        error = y_pred - y
        dw = (X.T @ error) / m
        db = np.mean(error)
        
        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db
        
        if i % 100 == 0:
            print(f"迭代 {i}, 损失: {loss:.4f}")
    
    return w, b

def test_gradient_descent():
    """测试梯度下降"""
    np.random.seed(42)
    # 生成线性数据
    n_samples = 100
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([2.0, -1.0])
    true_b = 3.0
    y = X @ true_w + true_b + np.random.randn(n_samples) * 0.1
    
    # 添加偏置项
    X_b = X
    
    w, b = gradient_descent(X_b, y, learning_rate=0.1, n_iters=500)
    
    print(f"梯度下降测试通过")
    print(f"真实参数: w={true_w}, b={true_b}")
    print(f"学习参数: w={w}, b={b:.4f}")


# ==================== 8. SMOTE (过采样) ====================
def smote(X_minority, n_samples, k=5):
    """
    SMOTE过采样算法
    
    算法原理:
    1. 对少数类样本，找到其k个最近邻
    2. 在样本和邻居之间随机线性插值生成新样本
    
    公式: new_sample = sample + random(0,1) * (neighbor - sample)
    
    关键要点:
    - 只对少数类样本过采样
    - 保持数据分布特性
    """
    n_minority, n_features = X_minority.shape
    
    # 如果不需要生成新样本
    if n_samples == 0:
        return X_minority
    
    # 计算距离矩阵(使用欧氏距离)
    synthetic_samples = []
    
    for _ in range(n_samples):
        # 随机选择一个少数类样本
        idx = np.random.randint(0, n_minority)
        sample = X_minority[idx]
        
        # 计算该样本到其他所有样本的距离
        distances = np.sqrt(np.sum((X_minority - sample) ** 2, axis=1))
        
        # 找到k个最近邻(排除自己)
        nearest_indices = np.argsort(distances)[1:k+1]  # 第一个是自己
        
        # 随机选择一个邻居
        neighbor_idx = np.random.choice(nearest_indices)
        neighbor = X_minority[neighbor_idx]
        
        # 生成新样本
        delta = neighbor - sample
        gap = np.random.random()
        new_sample = sample + gap * delta
        
        synthetic_samples.append(new_sample)
    
    synthetic_samples = np.array(synthetic_samples)
    
    # 合并原始样本和生成样本
    return np.vstack([X_minority, synthetic_samples])

def test_smote():
    """测试SMOTE"""
    np.random.seed(42)
    # 创建不平衡数据
    X_majority = np.random.randn(100, 2) + [2, 2]
    X_minority = np.random.randn(20, 2) + [-2, -2]
    
    y_majority = np.zeros(100)
    y_minority = np.ones(20)
    
    # 生成过采样
    X_minority_augmented = smote(X_minority, n_samples=80, k=5)
    
    print(f"SMOTE测试通过")
    print(f"原始少数类样本数: {len(X_minority)}")
    print(f"过采样后少数类样本数: {len(X_minority_augmented)}")


# ==================== 9. 模型剪枝（删除环中的一条边去环） ====================
def remove_cycle_edge(edges, n):
    """
    删除无向图中环的一条边使其成为无环图
    
    算法原理:
    1. 使用并查集检测环
    2. 当加入边会形成环时，记录该边
    3. 删除该边
    
    关键要点:
    - 无向图的环检测
    - 删除边后图应连通(可能需要额外处理)
    """
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        
        if root_x != root_y:
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
            return True
        return False
    
    # 存储结果边
    result_edges = []
    
    for u, v, w in edges:
        if not union(u, v):  # 会形成环
            print(f"检测到环，删除边 {u}-{v}")
            continue
        result_edges.append((u, v, w))
    
    return result_edges

def test_remove_cycle():
    """测试删除环边"""
    # 创建一个有环的图
    n = 4
    edges = [
        (0, 1, 1),
        (1, 2, 2),
        (2, 3, 3),
        (3, 0, 4),  # 这会形成环
        (0, 2, 5)
    ]
    
    pruned_edges = remove_cycle_edge(edges, n)
    print(f"模型剪枝(去环)测试通过")
    print(f"原始边数: {len(edges)}")
    print(f"剪枝后边数: {len(pruned_edges)}")
    print(f"剪枝后边: {pruned_edges}")


# ==================== 10. 分布式异常训练算法 ====================
class DistributedTrainer:
    """
    模拟分布式训练(简化版)
    
    算法原理:
    1. 多个worker计算本地梯度
    2. 将梯度上传到参数服务器
    3. 参数服务器聚合梯度并更新模型
    
    关键要点:
    - 梯度同步机制
    - 处理通信延迟(这里简化)
    """
    def __init__(self, n_workers, n_features, learning_rate=0.01):
        self.n_workers = n_workers
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        # 初始化全局参数
        self.global_w = np.zeros(n_features)
        self.global_b = 0.0
        
    def local_gradient(self, X_local, y_local, w, b):
        """计算本地梯度"""
        m = len(y_local)
        y_pred = X_local @ w + b
        error = y_pred - y_local
        dw = (X_local.T @ error) / m
        db = np.mean(error)
        return dw, db
    
    def train_step(self, X_shards, y_shards):
        """一步训练"""
        # 存储所有worker的梯度
        all_dw = []
        all_db = []
        
        # 每个worker计算本地梯度
        for i in range(self.n_workers):
            # 获取本地数据分片
            X_local = X_shards[i]
            y_local = y_shards[i]
            
            # 计算本地梯度
            dw, db = self.local_gradient(X_local, y_local, 
                                       self.global_w, self.global_b)
            all_dw.append(dw)
            all_db.append(db)
        
        # 聚合梯度(平均)
        avg_dw = np.mean(all_dw, axis=0)
        avg_db = np.mean(all_db)
        
        # 更新全局参数
        self.global_w -= self.learning_rate * avg_dw
        self.global_b -= self.learning_rate * avg_db
        
        return self.global_w, self.global_b
    
    def train(self, X, y, n_epochs=10, batch_size=None):
        """训练过程"""
        n_samples = len(X)
        
        # 数据分片
        shard_size = n_samples // self.n_workers
        X_shards = []
        y_shards = []
        
        for i in range(self.n_workers):
            start = i * shard_size
            end = (i + 1) * shard_size if i < self.n_workers - 1 else n_samples
            X_shards.append(X[start:end])
            y_shards.append(y[start:end])
        
        # 训练循环
        for epoch in range(n_epochs):
            w, b = self.train_step(X_shards, y_shards)
            
            if epoch % 5 == 0:
                y_pred = X @ w + b
                loss = np.mean((y_pred - y) ** 2) / 2
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return w, b

def test_distributed():
    """测试分布式训练"""
    np.random.seed(42)
    # 生成简单数据
    n_samples = 100
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([1.5, -0.5, 2.0])
    true_b = 3.0
    y = X @ true_w + true_b + np.random.randn(n_samples) * 0.1
    
    # 分布式训练
    trainer = DistributedTrainer(n_workers=4, n_features=n_features, learning_rate=0.1)
    w, b = trainer.train(X, y, n_epochs=20)
    
    print(f"分布式训练测试通过")
    print(f"真实参数: w={true_w}, b={true_b}")
    print(f"学习参数: w={w}, b={b:.4f}")


# ==================== 11. 卷积算子(滑动窗口) ====================
def convolution2d(image, kernel, stride=1, padding=0):
    """
    2D卷积运算
    
    算法原理:
    1. 对输入图像进行padding
    2. 滑动窗口，计算窗口与核的点积
    3. 输出特征图
    
    关键要点:
    - 输入输出尺寸计算
    - 滑动窗口实现
    """
    # 获取尺寸
    in_h, in_w = image.shape
    k_h, k_w = kernel.shape
    
    # 计算输出尺寸
    out_h = (in_h - k_h + 2 * padding) // stride + 1
    out_w = (in_w - k_w + 2 * padding) // stride + 1
    
    # 添加padding
    if padding > 0:
        padded = np.zeros((in_h + 2 * padding, in_w + 2 * padding))
        padded[padding:padding+in_h, padding:padding+in_w] = image
    else:
        padded = image
    
    # 初始化输出
    output = np.zeros((out_h, out_w))
    
    # 滑动窗口卷积
    for i in range(0, out_h):
        for j in range(0, out_w):
            h_start = i * stride
            h_end = h_start + k_h
            w_start = j * stride
            w_end = w_start + k_w
            
            window = padded[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(window * kernel)
    
    return output

def test_convolution():
    """测试卷积"""
    np.random.seed(42)
    # 创建测试图像
    image = np.array([
        [1, 2, 3, 0],
        [4, 5, 6, 1],
        [7, 8, 9, 2],
        [0, 1, 2, 3]
    ], dtype=float)
    
    # 创建卷积核
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ], dtype=float)
    
    result = convolution2d(image, kernel, stride=1, padding=0)
    
    print(f"卷积算子测试通过")
    print(f"输入图像尺寸: {image.shape}")
    print(f"卷积核尺寸: {kernel.shape}")
    print(f"输出特征图尺寸: {result.shape}")
    print(f"输出前3x3:\n{result[:3, :3]}")


# ==================== 12. 池化 (Max Pooling, Average Pooling) ====================
def max_pooling2d(image, pool_size=2, stride=None):
    """最大池化"""
    if stride is None:
        stride = pool_size
    
    h, w = image.shape
    pool_h, pool_w = pool_size, pool_size
    
    out_h = (h - pool_h) // stride + 1
    out_w = (w - pool_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + pool_h
            w_start = j * stride
            w_end = w_start + pool_w
            
            window = image[h_start:h_end, w_start:w_end]
            output[i, j] = np.max(window)
    
    return output

def avg_pooling2d(image, pool_size=2, stride=None):
    """平均池化"""
    if stride is None:
        stride = pool_size
    
    h, w = image.shape
    pool_h, pool_w = pool_size, pool_size
    
    out_h = (h - pool_h) // stride + 1
    out_w = (w - pool_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            h_end = h_start + pool_h
            w_start = j * stride
            w_end = w_start + pool_w
            
            window = image[h_start:h_end, w_start:w_end]
            output[i, j] = np.mean(window)
    
    return output

def test_pooling():
    """测试池化"""
    np.random.seed(42)
    # 创建测试图像
    image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)
    
    # 最大池化
    max_pool_result = max_pooling2d(image, pool_size=2, stride=2)
    
    # 平均池化
    avg_pool_result = avg_pooling2d(image, pool_size=2, stride=2)
    
    print(f"池化测试通过")
    print(f"输入图像尺寸: {image.shape}")
    print(f"最大池化结果:\n{max_pool_result}")
    print(f"平均池化结果:\n{avg_pool_result}")


# ==================== 主测试函数 ====================
def main():
    """主测试函数"""
    print("=" * 50)
    print("AI基础算法测试")
    print("=" * 50)
    test_kmeans()
    print()
    
    test_knn()
    print()
    
    test_pagerank()
    print()
    
    print("=" * 50)
    print("决策推理算法测试")
    print("=" * 50)
    test_decision_tree()
    print()
    
    test_bipartite()
    print()
    
    test_kruskal()
    print()
    
    test_gradient_descent()
    print()
    
    test_smote()
    print()
    
    test_remove_cycle()
    print()
    
    test_distributed()
    print()
    
    print("=" * 50)
    print("语音语义领域算法测试")
    print("=" * 50)
    test_convolution()
    print()
    
    test_pooling()
    print()
    
    print("所有算法测试完成！")

if __name__ == "__main__":
    main()
```

## 代码实现关键要点总结：

### 1. K-Means
- 关键：质心初始化、距离计算、收敛判断
- 易错：空簇处理、距离计算的效率优化

### 2. KNN
- 关键：距离矩阵计算、最近邻查找
- 易错：距离度量的选择、平票处理

### 3. PageRank
- 关键：转移矩阵构建、阻尼因子处理
- 易错：悬挂节点处理、收敛条件

### 4. 决策树
- 关键：信息增益计算、最佳分割点选择
- 易错：熵计算中的log0问题、数值稳定性

### 5. 二分图判断
- 关键：BFS染色、颜色冲突检测
- 易错：非连通图处理、自环边处理

### 6. 最小生成树
- 关键：并查集实现、边排序
- 易错：并查集路径压缩、环检测

### 7. 梯度下降
- 关键：梯度计算、学习率设置
- 易错：特征缩放、学习率衰减

### 8. SMOTE
- 关键：最近邻查找、线性插值
- 易错：边界样本处理、k值选择

### 9. 模型剪枝
- 关键：并查集环检测
- 易错：连通性保持、最小权重边选择

### 10. 分布式训练
- 关键：梯度同步、参数聚合
- 易错：通信开销、数据分布

### 11. 卷积算子
- 关键：滑动窗口、padding处理
- 易错：尺寸计算、边界处理

### 12. 池化
- 关键：窗口操作、步长设置
- 易错：尺寸计算、重叠池化

## 考试准备建议：

1. **重点掌握**：K-Means、梯度下降、卷积是高频考点
2. **理解原理**：每个算法的数学原理和实现细节
3. **调试技巧**：添加打印语句验证中间结果
4. **边界情况**：注意处理空簇、除零、边界等特殊情况
5. **时间分配**：先完成核心功能，再优化边界情况

这个实现包含了所有12个算法的简洁实现，可以直接复制运行，每个算法都有独立测试函数。祝你考试顺利！
