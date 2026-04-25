# AISchool — AI 算法考试速成代码库

一天速成 10 个核心 AI / 数据科学 / 图论算法。包含精简实现、学习指南、练习题和模拟考试。

## 项目结构

```
AISchool/
├── README.md                    ← 你在这里
├── requirements.txt             ← 依赖 (仅 numpy)
├── run_all.py                   ← 一键运行所有算法示例
│
├── algorithms/                  ← 10 个算法的核心实现（含详细注释）
│   ├── 01_kmeans.py                K-Means 聚类
│   ├── 02_knn.py                   KNN 分类
│   ├── 03_pagerank.py              PageRank
│   ├── 04_decision_tree.py         决策树 (ID3)
│   ├── 05_convolution.py           2D 卷积
│   ├── 06_max_pooling.py           最大池化
│   ├── 07_bipartite_check.py       二分图判断
│   ├── 08_kruskal.py               Kruskal 最小生成树
│   ├── 09_gradient_descent.py      梯度下降
│   └── 10_smote.py                 SMOTE 过采样
│
├── docs/                        ← 学习文档
│   ├── study_guide.md              一天学习计划 + 记忆卡片
│   ├── cheatsheet.md               考试速查表（考前 30 分钟用）
│   └── sprint_plan.md              考前冲刺计划（1天/半天/1小时方案）
│
└── exercises/                   ← 练习与考试
    ├── 01_fill_in_the_blank.py     填空练习（10 题）
    └── 02_mock_exam.py             模拟考试（10 题，100 分）
```

## 快速开始

```bash
pip install numpy

# 运行全部算法示例
python3 run_all.py

# 运行单个算法
python3 algorithms/01_kmeans.py

# 运行指定算法
python3 run_all.py 01 05 09

# 做填空练习
python3 exercises/01_fill_in_the_blank.py

# 做模拟考试
python3 exercises/02_mock_exam.py
```

## 算法列表

| # | 算法 | 核心概念 | 口诀 |
|---|------|----------|------|
| 01 | K-Means | 簇心迭代 | 初始化→分配→更新→重复 |
| 02 | KNN | 欧氏距离+投票 | 算距离→排序→投票 |
| 03 | PageRank | 幂迭代 | 转移矩阵→幂迭代→归一化 |
| 04 | 决策树 | 信息增益 | 算熵→算IG→选特征→递归 |
| 05 | 2D 卷积 | 滑动窗口 | 双层循环→提窗口→乘→求和 |
| 06 | Max Pooling | 下采样 | 双层循环→提窗口→取max |
| 07 | 二分图 | BFS染色 | BFS→染色→查冲突 |
| 08 | Kruskal MST | 并查集 | 排序→并查集→逐边加 |
| 09 | 梯度下降 | 线性回归 | 预测→误差→梯度→更新 |
| 10 | SMOTE | 过采样插值 | 选样本→找近邻→插值 |

## 学习路线

### 第一步：阅读学习指南
→ [`docs/study_guide.md`](docs/study_guide.md) — 包含一天学习计划和 10 个算法的记忆卡片

### 第二步：逐个学习算法
→ `algorithms/` 目录下每个文件都包含：算法背景、流程图、核心公式、记忆要点、易错点、可运行示例

### 第三步：做填空练习
→ [`exercises/01_fill_in_the_blank.py`](exercises/01_fill_in_the_blank.py) — 用 `_____` 标记关键代码，填完运行即自动判分

### 第四步：模拟考试
→ [`exercises/02_mock_exam.py`](exercises/02_mock_exam.py) — 从零实现 10 个算法，100 分制自动判分

### 第五步：考前速查
→ [`docs/cheatsheet.md`](docs/cheatsheet.md) — 公式、对比、复杂度一页纸

### 灵活安排
→ [`docs/sprint_plan.md`](docs/sprint_plan.md) — 根据剩余时间选择不同冲刺方案
