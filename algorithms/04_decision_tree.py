"""
决策树核心 —— 基于信息增益 (ID3)
================================

【算法背景】
决策树是一种经典的监督学习算法，由 Ross Quinlan 在 1986 年提出 ID3 版本。
它通过递归地选择最优特征来分裂数据，构建一棵树形分类器。
直觉上就像玩"20个问题"游戏 — 每次问一个最有区分度的问题。

【适用场景】
- 医疗诊断、信贷审批、客户流失预测
- 结果可解释性要求高的场景（决策树天然可解释）

【核心概念详解】

  1. 信息熵 (Entropy):
     衡量数据集的"混乱程度"
     H(S) = -Σ pᵢ * log₂(pᵢ)
     - 纯净集合 (全是同一类): H = 0     (完全确定)
     - 均匀分布 (两类各一半): H = 1     (最不确定)

     例: 10个样本中 7正3负 → H = -0.7*log₂(0.7) - 0.3*log₂(0.3) ≈ 0.881

  2. 信息增益 (Information Gain):
     用某个特征分裂后，熵减少了多少
     IG(S, A) = H(S) - Σ (|Sᵥ|/|S|) * H(Sᵥ)
     - H(S) 是分裂前的熵
     - Sᵥ 是按特征 A 的值 v 分裂后的子集
     - 信息增益越大，说明这个特征越"有用"

【算法流程（递归）】
    ┌──────────────────────────────┐
    │ 终止条件检查:                  │
    │   - 所有标签相同 → 返回该标签   │
    │   - 没有特征可分 → 返回多数类   │
    │   - 达到最大深度 → 返回多数类   │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 1: 计算每个特征的信息增益   │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: 选信息增益最大的特征    │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 3: 按该特征的每个值分裂    │
    │   数据集，递归构建子树          │
    └──────────────────────────────┘

【口诀】算熵 → 算信息增益 → 选最优特征 → 递归

【记忆要点 - 考试必背】
  1. 熵的计算: H = -Σ (c/total) * log₂(c/total)
     ★ 用 Counter 统计每类数量，除以总数得概率，代入公式
  2. 信息增益 = 父节点熵 - 加权子节点熵
     ★ 权重 = 子集大小 / 父集大小
  3. 递归三个终止条件必须记住:
     - len(set(labels)) == 1      → 纯净
     - X.shape[1] == 0            → 没特征了
     - depth >= max_depth          → 到达深度限制
  4. 分裂后要 np.delete 删掉已用的特征列

【易错点】
  - log₂ 不是 ln！用 np.log2 而不是 np.log
  - 信息增益是"减少量"：父熵 - 子熵加权和（不是反过来）
  - 递归分裂时，已经用过的特征要从矩阵中删除
  - ID3 只能处理离散特征；连续特征需要先离散化

【三种决策树算法对比（常考）】
  ID3:  用信息增益 → 偏好取值多的特征
  C4.5: 用信息增益率 → 修正 ID3 的偏好
  CART: 用基尼系数 → 二叉树，可做回归

【复杂度】
  构建: O(n * m * log(n))   n=样本数, m=特征数
  预测: O(树深度)           通常 O(log(n))
"""
import numpy as np
from collections import Counter


def entropy(labels):
    """
    计算信息熵 H(S) = -Σ pᵢ * log₂(pᵢ)
    labels 为空时返回 0
    """
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    total = len(labels)
    return -sum(
        (c / total) * np.log2(c / total)
        for c in counts.values()
    )


def information_gain(X_column, labels):
    """
    计算某个特征列的信息增益
    IG = H(parent) - Σ (|child|/|parent|) * H(child)
    """
    parent_entropy = entropy(labels)

    values = set(X_column)
    weighted_child_entropy = 0
    for val in values:
        mask = X_column == val
        child_labels = labels[mask]
        weight = len(child_labels) / len(labels)
        weighted_child_entropy += weight * entropy(child_labels)

    return parent_entropy - weighted_child_entropy


def build_tree(X, labels, feature_names=None, depth=0, max_depth=5):
    """
    递归构建决策树，返回嵌套字典表示的树结构
    """
    # 终止条件 1: 所有标签相同
    if len(set(labels)) == 1:
        return labels[0]
    # 终止条件 2&3: 没有特征可分 或 达到最大深度
    if X.shape[1] == 0 or depth >= max_depth:
        return Counter(labels).most_common(1)[0][0]

    # 计算每个特征的信息增益，选最大的
    gains = [information_gain(X[:, i], labels) for i in range(X.shape[1])]
    best_feature = np.argmax(gains)

    if feature_names:
        feature_name = feature_names[best_feature]
    else:
        feature_name = f"feature_{best_feature}"

    # 按最优特征的值分裂，递归构建子树
    tree = {feature_name: {}}
    for val in sorted(set(X[:, best_feature])):
        mask = X[:, best_feature] == val
        sub_X = np.delete(X[mask], best_feature, axis=1)
        sub_names = (
            feature_names[:best_feature] + feature_names[best_feature + 1:]
            if feature_names else None
        )
        tree[feature_name][val] = build_tree(
            sub_X, labels[mask], sub_names, depth + 1, max_depth
        )

    return tree


def predict_one(tree, sample, feature_names):
    """用构建好的树对单个样本做预测"""
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    idx = feature_names.index(feature)
    value = sample[idx]
    subtree = tree[feature].get(value, None)
    if subtree is None:
        return None
    return predict_one(subtree, sample, feature_names)


# ========== 运行示例 ==========
if __name__ == "__main__":
    # 经典「打网球」数据集
    # 特征: 天气(0晴/1阴/2雨), 温度(0热/1适中/2冷), 湿度(0高/1正常), 风(0弱/1强)
    X = np.array([
        [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [2, 1, 0, 0],
        [2, 2, 1, 0], [2, 2, 1, 1], [1, 2, 1, 1], [0, 1, 0, 0],
        [0, 2, 1, 0], [2, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 1],
        [1, 0, 1, 0], [2, 1, 0, 1],
    ])
    labels = np.array(["否", "否", "是", "是", "是", "否", "是",
                        "否", "是", "是", "是", "是", "是", "否"])
    names = ["天气", "温度", "湿度", "风力"]

    tree = build_tree(X, labels, names)

    import json
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys(v) for k, v in obj.items()}
        return obj if not isinstance(obj, np.integer) else int(obj)

    print("决策树结构:")
    print("=" * 40)
    print(json.dumps(convert_keys(tree), ensure_ascii=False, indent=2))

    print("\n--- 理解辅助：信息增益计算过程 ---")
    print(f"  总体熵 H = {entropy(labels):.4f}")
    for i, name in enumerate(names):
        ig = information_gain(X[:, i], labels)
        print(f"  特征 '{name}' 的信息增益 IG = {ig:.4f}")
    print(f"  → 选择信息增益最大的特征 '{names[np.argmax([information_gain(X[:, i], labels) for i in range(4)])]}' 作为根节点")

    print("\n预测测试:")
    test = np.array([0, 1, 0, 0])  # 晴, 适中, 高湿度, 弱风
    print(f"  输入: 天气=晴, 温度=适中, 湿度=高, 风力=弱 → 预测: {predict_one(tree, test, names)}")
