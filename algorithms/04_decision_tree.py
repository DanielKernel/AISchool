"""
决策树核心 —— 基于信息增益 (ID3)
================================
核心思想：计算每个特征的信息增益 → 选增益最大的特征做分裂 → 递归构建子树

口诀：算熵 → 算信息增益 → 选最优特征 → 递归
"""
import numpy as np
from collections import Counter


def entropy(labels):
    """计算信息熵 H(S) = -Σ p_i * log2(p_i)"""
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

    # 按特征值分组
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

    参数:
        X: 特征矩阵, shape (n_samples, n_features)
        labels: 标签数组
        feature_names: 特征名称列表
        depth: 当前深度
        max_depth: 最大深度
    返回:
        树结构 (字典 或 叶节点标签)
    """
    # 终止条件: 所有标签相同、达到最大深度、没有特征可分
    if len(set(labels)) == 1:
        return labels[0]
    if X.shape[1] == 0 or depth >= max_depth:
        return Counter(labels).most_common(1)[0][0]

    # 计算每个特征的信息增益，选最大的
    gains = [information_gain(X[:, i], labels) for i in range(X.shape[1])]
    best_feature = np.argmax(gains)

    if feature_names:
        feature_name = feature_names[best_feature]
    else:
        feature_name = f"feature_{best_feature}"

    # 按最优特征的值分裂
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
    print("决策树结构:")
    import json
    def convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys(v) for k, v in obj.items()}
        return obj if not isinstance(obj, np.integer) else int(obj)
    print(json.dumps(convert_keys(tree), ensure_ascii=False, indent=2))

    print("\n预测测试:")
    test = np.array([0, 1, 0, 0])  # 晴, 适中, 高湿度, 弱风
    print(f"  输入: 天气=晴, 温度=适中, 湿度=高, 风力=弱 → 预测: {predict_one(tree, test, names)}")
