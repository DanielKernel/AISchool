"""
填空练习题
==========

使用方法:
    python3 exercises/01_fill_in_the_blank.py

每个函数中有 _____ 标记的位置需要你填写正确代码。
填完后运行本文件，会自动检查你的答案是否正确。

提示: 先自己思考，实在想不出再去 algorithms/ 目录下看对应的完整实现。
"""
import numpy as np
from collections import Counter, deque

# ================================================================
# 练习 1: K-Means 聚类
# ================================================================

def exercise_kmeans(X, k, max_iters=100):
    """
    提示: 口诀 — 初始化 → 分配 → 更新 → 重复
    """
    n = X.shape[0]

    # TODO 1.1: 随机选 k 个不重复的样本索引作为初始簇心
    indices = _____  # np.random.choice(???)
    centroids = X[indices].copy()

    for _ in range(max_iters):
        # TODO 1.2: 计算每个点到每个簇心的欧氏距离
        # 提示: X[:, np.newaxis] 形状 (n,1,d), centroids 形状 (k,d)
        distances = _____  # np.linalg.norm(???, axis=???)

        # TODO 1.3: 每个点分配到距离最近的簇心
        labels = _____  # np.argmin(???, axis=???)

        # TODO 1.4: 更新簇心为簇内均值
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# ================================================================
# 练习 2: KNN 分类
# ================================================================

def exercise_knn(X_train, y_train, x_query, k=3):
    """
    提示: 口诀 — 算距离 → 排序 → 投票
    """
    # TODO 2.1: 计算查询点到所有训练样本的欧氏距离
    distances = _____  # np.linalg.norm(???, axis=???)

    # TODO 2.2: 找距离最小的 k 个样本索引
    k_nearest_indices = _____  # np.argsort(???)[:???]

    # TODO 2.3: 多数投票
    k_nearest_labels = y_train[k_nearest_indices]
    most_common = _____  # Counter(???).most_common(???)

    return most_common[0][0]


# ================================================================
# 练习 3: PageRank
# ================================================================

def exercise_pagerank(adj_matrix, damping=0.85, max_iters=100, tol=1e-6):
    """
    提示: 口诀 — 构建转移矩阵 → 幂迭代 → 归一化
    """
    n = adj_matrix.shape[0]

    # TODO 3.1: 构建转移矩阵（行归一化后转置）
    out_degree = adj_matrix.sum(axis=1, keepdims=True)
    out_degree[out_degree == 0] = 1
    M = _____  # (adj_matrix / ???).???

    # TODO 3.2: 初始化排名向量
    ranks = _____  # np.ones(???) / ???

    for _ in range(max_iters):
        # TODO 3.3: 幂迭代核心公式 ★
        new_ranks = _____  # (1 - damping) / ??? + damping * ??? @ ???

        if np.linalg.norm(new_ranks - ranks) < tol:
            break
        ranks = new_ranks

    return ranks


# ================================================================
# 练习 4: 信息熵与信息增益
# ================================================================

def exercise_entropy(labels):
    """
    提示: H = -Σ (c/total) * log₂(c/total)
    """
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    total = len(labels)
    # TODO 4.1: 计算信息熵
    return _____  # -sum(??? for c in counts.values())


def exercise_information_gain(X_column, labels):
    """
    提示: IG = H(parent) - Σ weight * H(child)
    """
    parent_entropy = exercise_entropy(labels)

    values = set(X_column)
    weighted_child_entropy = 0
    for val in values:
        mask = X_column == val
        child_labels = labels[mask]
        # TODO 4.2: 计算权重和子节点熵
        weight = _____  # len(???) / len(???)
        weighted_child_entropy += weight * exercise_entropy(child_labels)

    # TODO 4.3: 返回信息增益
    return _____  # ??? - ???


# ================================================================
# 练习 5: 2D 卷积
# ================================================================

def exercise_conv2d(image, kernel, stride=1, padding=0):
    """
    提示: 口诀 — 双层循环 → 提取窗口 → 逐元素乘 → 求和
    """
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    H, W = image.shape
    kH, kW = kernel.shape

    # TODO 5.1: 计算输出尺寸 ★ 必背公式
    out_H = _____  # (??? - ??? ) // ??? + 1
    out_W = _____  # (??? - ??? ) // ??? + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # TODO 5.2: 提取窗口区域
            region = _____  # image[???:???, ???:???]
            # TODO 5.3: 卷积计算
            output[i, j] = _____  # np.sum(??? * ???)

    return output


# ================================================================
# 练习 6: Max Pooling
# ================================================================

def exercise_max_pooling(image, pool_size=2, stride=2):
    """
    提示: 和卷积框架一样，只是窗口操作不同
    """
    H, W = image.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            # TODO 6.1: 取窗口内最大值
            output[i, j] = _____  # np.???(region)

    return output


# ================================================================
# 练习 7: 二分图判断
# ================================================================

def exercise_is_bipartite(graph):
    """
    提示: 口诀 — BFS → 染色 → 检查冲突
    """
    color = {}

    for start_node in graph:
        if start_node in color:
            continue

        queue = deque([start_node])
        color[start_node] = 0

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    # TODO 7.1: 相邻节点染相反颜色
                    color[neighbor] = _____  # ??? - color[???]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    # TODO 7.2: 冲突时返回什么？
                    return _____

    return True


# ================================================================
# 练习 8: Kruskal 最小生成树
# ================================================================

def exercise_kruskal(n, edges):
    """
    提示: 口诀 — 排序边 → 并查集 → 逐边加入
    """
    parent = list(range(n))

    def find(x):
        # TODO 8.1: 带路径压缩的 find
        if parent[x] != x:
            parent[x] = _____  # find(???)
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        # TODO 8.2: 判断是否在同一集合
        if rx == ry:
            return _____
        parent[ry] = rx
        return True

    # TODO 8.3: 按权重排序
    edges_sorted = _____  # sorted(edges, key=lambda e: ???)

    mst = []
    total = 0
    for weight, u, v in edges_sorted:
        if union(u, v):
            mst.append((u, v, weight))
            total += weight
            # TODO 8.4: 何时停止?
            if len(mst) == _____:  # ??? - ???
                break

    return mst, total


# ================================================================
# 练习 9: 梯度下降
# ================================================================

def exercise_gradient_descent(X, y, lr=0.01, max_iters=1000, tol=1e-6):
    """
    提示: 口诀 — 算预测 → 算误差 → 算梯度 → 更新参数
    """
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    losses = []

    for _ in range(max_iters):
        # TODO 9.1: 计算预测值
        y_pred = _____  # ??? @ ???

        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)

        # TODO 9.2: 计算梯度 ★
        gradient = _____  # (2 / ???) * (???.T @ ???)

        # TODO 9.3: 更新参数（注意方向！）
        w = _____  # ??? ??? lr * gradient

        if len(losses) > 1 and abs(losses[-2] - losses[-1]) < tol:
            break

    return w, losses


# ================================================================
# 练习 10: SMOTE 过采样
# ================================================================

def exercise_smote(X_minority, n_new_samples, k=5):
    """
    提示: 口诀 — 找少数类 → 找K近邻 → 随机插值
    """
    n_samples, n_features = X_minority.shape
    k = min(k, n_samples - 1)
    synthetic = np.zeros((n_new_samples, n_features))

    for i in range(n_new_samples):
        idx = np.random.randint(0, n_samples)
        sample = X_minority[idx]

        distances = np.linalg.norm(X_minority - sample, axis=1)
        # TODO 10.1: 找 K 个近邻（跳过自身）
        neighbor_indices = _____  # np.argsort(???)[???:???]

        nn_idx = neighbor_indices[np.random.randint(0, len(neighbor_indices))]
        neighbor = X_minority[nn_idx]

        # TODO 10.2: 插值公式 ★
        lam = np.random.random()
        synthetic[i] = _____  # sample + ??? * (??? - ???)

    return synthetic


# ================================================================
# 答案检查器
# ================================================================

def _check(name, test_func):
    try:
        test_func()
        print(f"  ✓ {name}")
        return True
    except Exception as e:
        err = str(e)
        if "_____" in err or "NameError" in type(e).__name__:
            print(f"  ○ {name} — 尚未填写")
        else:
            print(f"  ✗ {name} — {type(e).__name__}: {err}")
        return False


def run_checks():
    print("=" * 50)
    print("填空练习自动检查")
    print("=" * 50)
    passed = 0
    total = 10

    np.random.seed(42)

    # 1. K-Means
    def test_kmeans():
        data = np.vstack([np.random.randn(20, 2) + [0, 0],
                          np.random.randn(20, 2) + [5, 5]])
        c, l = exercise_kmeans(data, k=2)
        assert c.shape == (2, 2) and len(set(l)) == 2
    passed += _check("练习1: K-Means", test_kmeans)

    # 2. KNN
    def test_knn():
        X = np.array([[0,0],[0,1],[1,0],[5,5],[5,6],[6,5]], dtype=float)
        y = np.array([0,0,0,1,1,1])
        assert exercise_knn(X, y, np.array([0.5, 0.5]), k=3) == 0
        assert exercise_knn(X, y, np.array([5.5, 5.5]), k=3) == 1
    passed += _check("练习2: KNN", test_knn)

    # 3. PageRank
    def test_pagerank():
        adj = np.array([[0,1,1,0],[0,0,1,0],[1,0,0,0],[0,0,1,0]], dtype=float)
        r = exercise_pagerank(adj)
        assert r.shape == (4,) and abs(r.sum() - 1.0) < 0.01
        assert np.argmax(r) == 2
    passed += _check("练习3: PageRank", test_pagerank)

    # 4. 信息熵与增益
    def test_entropy():
        assert abs(exercise_entropy(np.array([0,0,1,1])) - 1.0) < 0.01
        assert abs(exercise_entropy(np.array([0,0,0,0])) - 0.0) < 0.01
        X_col = np.array([0,0,1,1,1,0])
        labels = np.array([0,0,1,1,1,0])
        ig = exercise_information_gain(X_col, labels)
        assert ig > 0
    passed += _check("练习4: 信息熵与增益", test_entropy)

    # 5. 卷积
    def test_conv():
        img = np.ones((4,4))
        k = np.ones((2,2))
        r = exercise_conv2d(img, k)
        assert r.shape == (3,3) and np.allclose(r, 4.0)
    passed += _check("练习5: 卷积", test_conv)

    # 6. 池化
    def test_pool():
        img = np.array([[1,3],[5,6]], dtype=float)
        r = exercise_max_pooling(img, 2, 2)
        assert r.shape == (1,1) and r[0,0] == 6.0
    passed += _check("练习6: Max Pooling", test_pool)

    # 7. 二分图
    def test_bipartite():
        assert exercise_is_bipartite({0:[1,3],1:[0,2],2:[1,3],3:[2,0]}) == True
        assert exercise_is_bipartite({0:[1,2],1:[0,2],2:[0,1]}) == False
    passed += _check("练习7: 二分图判断", test_bipartite)

    # 8. Kruskal
    def test_kruskal():
        edges = [(1,0,1),(2,1,3),(3,0,3),(4,1,2),(6,3,4)]
        mst, total = exercise_kruskal(5, edges)
        assert len(mst) == 4
    passed += _check("练习8: Kruskal MST", test_kruskal)

    # 9. 梯度下降
    def test_gd():
        x = np.linspace(0, 5, 20)
        y = 2 * x + 1
        X = np.column_stack([np.ones_like(x), x])
        w, losses = exercise_gradient_descent(X, y, lr=0.01, max_iters=2000)
        assert abs(w[0] - 1) < 0.5 and abs(w[1] - 2) < 0.5
    passed += _check("练习9: 梯度下降", test_gd)

    # 10. SMOTE
    def test_smote():
        Xm = np.random.randn(10, 2)
        s = exercise_smote(Xm, 5, k=3)
        assert s.shape == (5, 2)
    passed += _check("练习10: SMOTE", test_smote)

    print()
    print(f"通过: {passed}/{total}")
    if passed == total:
        print("🎉 全部通过！你已经掌握了所有算法的核心代码！")
    elif passed >= 7:
        print("👍 大部分通过，再补强薄弱点就可以了！")
    elif passed >= 4:
        print("📖 还需要多练习，建议重新阅读对应的 algorithms/ 文件。")
    else:
        print("💪 加油！建议从 docs/study_guide.md 开始系统学习。")


if __name__ == "__main__":
    run_checks()
