"""
模拟考试
========

使用方法:
    python3 exercises/02_mock_exam.py

本文件包含 10 道模拟考试题，覆盖全部算法。
每道题要求你**从零实现**对应的函数（不是填空）。

建议:
    1. 先自己独立完成，严格限时（每题 8 分钟）
    2. 完成后运行本文件检查答案
    3. 错题回去看 algorithms/ 目录下对应文件
"""
import numpy as np
from collections import Counter, deque


# ================================================================
# 第 1 题: 实现 K-Means 聚类 (15 分)
# ================================================================
# 要求: 接收数据矩阵 X 和簇数 k，返回 (centroids, labels)

def exam_kmeans(X, k, max_iters=100):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 2 题: 实现 KNN 分类预测 (10 分)
# ================================================================
# 要求: 对单个查询点 x_query 返回预测标签

def exam_knn(X_train, y_train, x_query, k=3):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 3 题: 实现 PageRank (15 分)
# ================================================================
# 要求: 接收邻接矩阵，返回 PageRank 向量

def exam_pagerank(adj_matrix, damping=0.85, max_iters=100, tol=1e-6):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 4 题: 实现信息熵和信息增益 (10 分)
# ================================================================

def exam_entropy(labels):
    # ---- 在此作答 ----
    pass


def exam_information_gain(X_column, labels):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 5 题: 实现 2D 卷积 (10 分)
# ================================================================
# 要求: 支持 stride 和 padding 参数

def exam_conv2d(image, kernel, stride=1, padding=0):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 6 题: 实现 Max Pooling (5 分)
# ================================================================

def exam_max_pooling(image, pool_size=2, stride=2):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 7 题: 实现二分图判断 (10 分)
# ================================================================
# 要求: 用 BFS 染色法，处理多连通分量

def exam_is_bipartite(graph):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 8 题: 实现 Kruskal 最小生成树 (15 分)
# ================================================================
# 要求: 含并查集（路径压缩），返回 (mst_edges, total_weight)

def exam_kruskal(n, edges):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 9 题: 实现梯度下降 (线性回归) (5 分)
# ================================================================
# 要求: 返回 (w, losses)

def exam_gradient_descent(X, y, lr=0.01, max_iters=1000, tol=1e-6):
    # ---- 在此作答 ----
    pass


# ================================================================
# 第 10 题: 实现 SMOTE 过采样 (5 分)
# ================================================================
# 要求: 返回 n_new_samples 个合成样本

def exam_smote(X_minority, n_new_samples, k=5):
    # ---- 在此作答 ----
    pass


# ================================================================
# 自动判分
# ================================================================

def _test(name, score, func):
    try:
        func()
        print(f"  ✓ {name} ({score}分)")
        return score
    except Exception as e:
        err_str = str(e)
        err_type = type(e).__name__
        is_unanswered = (
            "NoneType" in err_str
            or "NoneType" in err_type
            or err_str == ""
            or err_str == "None"
        )
        if is_unanswered:
            print(f"  ○ {name} ({score}分) — 尚未作答")
        else:
            print(f"  ✗ {name} ({score}分) — {err_type}: {err_str}")
        return 0


def grade():
    print("=" * 55)
    print("模拟考试 — 自动判分")
    print("=" * 55)
    total = 0
    full = 100
    np.random.seed(42)

    # Q1: K-Means
    def t1():
        data = np.vstack([np.random.randn(30, 2), np.random.randn(30, 2) + 5])
        c, l = exam_kmeans(data, 2)
        assert c.shape == (2, 2), f"簇心形状应为(2,2)，得到{c.shape}"
        assert len(set(l)) == 2, "应有2个簇"
    total += _test("第1题: K-Means", 15, t1)

    # Q2: KNN
    def t2():
        X = np.array([[0,0],[0,1],[1,0],[5,5],[5,6],[6,5]], dtype=float)
        y = np.array([0,0,0,1,1,1])
        assert exam_knn(X, y, np.array([0.5, 0.5]), 3) == 0
        assert exam_knn(X, y, np.array([5.5, 5.5]), 3) == 1
    total += _test("第2题: KNN", 10, t2)

    # Q3: PageRank
    def t3():
        adj = np.array([[0,1,1,0],[0,0,1,0],[1,0,0,0],[0,0,1,0]], dtype=float)
        r = exam_pagerank(adj)
        assert r.shape == (4,), f"应返回长度4的向量，得到{r.shape}"
        assert abs(r.sum() - 1.0) < 0.05, f"PageRank 和应约为1，得到{r.sum()}"
        assert np.argmax(r) == 2, f"页面2应排名最高"
    total += _test("第3题: PageRank", 15, t3)

    # Q4: 信息熵与增益
    def t4():
        assert abs(exam_entropy(np.array([0,0,1,1])) - 1.0) < 0.01
        assert abs(exam_entropy(np.array([0,0,0])) - 0.0) < 0.01
        X_col = np.array([0,0,1,1])
        labels = np.array([0,0,1,1])
        ig = exam_information_gain(X_col, labels)
        assert abs(ig - 1.0) < 0.01, f"完全分离时IG应为1.0，得到{ig}"
    total += _test("第4题: 信息熵与增益", 10, t4)

    # Q5: 卷积
    def t5():
        img = np.ones((4, 4))
        k = np.ones((2, 2))
        r = exam_conv2d(img, k)
        assert r.shape == (3, 3), f"输出尺寸应为(3,3)，得到{r.shape}"
        assert np.allclose(r, 4.0), f"全1图像用全1核卷积应得4"
        r2 = exam_conv2d(img, k, padding=1)
        assert r2.shape == (5, 5), f"padding=1时输出应为(5,5)，得到{r2.shape}"
    total += _test("第5题: 卷积", 10, t5)

    # Q6: Max Pooling
    def t6():
        img = np.array([[1,3,2,4],[5,6,1,2],[7,2,3,8],[4,9,1,5]], dtype=float)
        r = exam_max_pooling(img, 2, 2)
        assert r.shape == (2, 2), f"输出尺寸应为(2,2)，得到{r.shape}"
        assert np.allclose(r, [[6,4],[9,8]])
    total += _test("第6题: Max Pooling", 5, t6)

    # Q7: 二分图
    def t7():
        assert exam_is_bipartite({0:[1,3],1:[0,2],2:[1,3],3:[2,0]}) == True
        assert exam_is_bipartite({0:[1,2],1:[0,2],2:[0,1]}) == False
        assert exam_is_bipartite({0:[2,3],1:[2,3],2:[0,1],3:[0,1]}) == True
    total += _test("第7题: 二分图判断", 10, t7)

    # Q8: Kruskal
    def t8():
        edges = [(1,0,1),(2,1,3),(3,0,3),(4,1,2),(5,2,3),(7,2,4),(6,3,4)]
        mst, w = exam_kruskal(5, edges)
        assert len(mst) == 4, f"MST应有4条边，得到{len(mst)}"
        assert w == 13, f"总权重应为13，得到{w}"
    total += _test("第8题: Kruskal MST", 15, t8)

    # Q9: 梯度下降
    def t9():
        x = np.linspace(0, 5, 30)
        y = 2 * x + 1
        X = np.column_stack([np.ones_like(x), x])
        w, losses = exam_gradient_descent(X, y, lr=0.01, max_iters=3000)
        assert abs(w[0] - 1) < 0.5, f"截距应约为1，得到{w[0]:.2f}"
        assert abs(w[1] - 2) < 0.5, f"斜率应约为2，得到{w[1]:.2f}"
        assert losses[-1] < losses[0], "损失应该下降"
    total += _test("第9题: 梯度下降", 5, t9)

    # Q10: SMOTE
    def t10():
        Xm = np.random.randn(10, 3)
        s = exam_smote(Xm, 20, k=3)
        assert s.shape == (20, 3), f"输出形状应为(20,3)，得到{s.shape}"
        for i in range(20):
            dists = np.linalg.norm(Xm - s[i], axis=1)
            assert dists.min() < np.linalg.norm(Xm.max(0) - Xm.min(0)), \
                "合成样本应在原始样本附近"
    total += _test("第10题: SMOTE", 5, t10)

    print()
    print(f"总分: {total}/{full}")
    if total == full:
        print("🎉 满分！考试一定没问题！")
    elif total >= 80:
        print("👍 优秀！再巩固一下薄弱项就完美了。")
    elif total >= 60:
        print("📖 及格了，但建议回看错题对应的算法文件再练一遍。")
    elif total >= 40:
        print("⚠️  还需要更多练习，重点看 docs/study_guide.md 的记忆卡片。")
    else:
        print("💪 不要灰心，从 algorithms/ 目录一个一个学起！")


if __name__ == "__main__":
    grade()
