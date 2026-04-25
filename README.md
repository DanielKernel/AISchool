# AISchool — AI 算法速成代码库

一天速成 **10 个** 核心 AI / 数据科学 / 图论算法的 Python 实现。实现集中在 [`aischool/algorithm_core.py`](aischool/algorithm_core.py)（单一源码，避免与根目录、`algorithms/` 重复）；[`algorithms/*.py`](algorithms/) 保留详细注释与独立运行示例。

## 快速开始

```bash
pip install numpy

# 运行单个算法示例
python algorithms/01_kmeans.py

# 一键运行根目录分主题演示
python run_all.py

# 自检（assert）
python examples/algorithms_reference.py
```

## 算法列表

| # | 算法 | 核心概念 | 文件 |
|---|------|----------|------|
| 01 | K-Means 聚类 | 簇心迭代 | [`algorithms/01_kmeans.py`](algorithms/01_kmeans.py) |
| 02 | KNN 分类 | 欧氏距离 + 投票 | [`algorithms/02_knn.py`](algorithms/02_knn.py) |
| 03 | PageRank | 幂迭代 | [`algorithms/03_pagerank.py`](algorithms/03_pagerank.py) |
| 04 | 决策树 (ID3) | 信息增益 | [`algorithms/04_decision_tree.py`](algorithms/04_decision_tree.py) |
| 05 | 2D 卷积 | 滑动窗口 | [`algorithms/05_convolution.py`](algorithms/05_convolution.py) |
| 06 | 最大池化 | Max Pooling | [`algorithms/06_max_pooling.py`](algorithms/06_max_pooling.py) |
| 07 | 二分图判断 | BFS 染色 | [`algorithms/07_bipartite_check.py`](algorithms/07_bipartite_check.py) |
| 08 | 最小生成树 | Kruskal + 并查集 | [`algorithms/08_kruskal.py`](algorithms/08_kruskal.py) |
| 09 | 梯度下降 | 线性回归 | [`algorithms/09_gradient_descent.py`](algorithms/09_gradient_descent.py) |
| 10 | SMOTE | 过采样插值 | [`algorithms/10_smote.py`](algorithms/10_smote.py) |

## 学习指南

详细的一日计划、**如何加强练习**、闭卷模拟考与考试技巧见 [**STUDY_GUIDE.md**](STUDY_GUIDE.md)。`docs/` 下仅保留指向该文件的索引。
