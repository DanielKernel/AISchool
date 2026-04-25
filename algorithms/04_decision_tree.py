"""
决策树核心 —— 基于信息增益 (ID3)；熵与增益见 aischool.algorithm_core。
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import numpy as np
from collections import Counter

from aischool.algorithm_core import entropy, information_gain


def build_tree(X, labels, feature_names=None, depth=0, max_depth=5):
    if len(set(labels)) == 1:
        return labels[0]
    if X.shape[1] == 0 or depth >= max_depth:
        return Counter(labels).most_common(1)[0][0]

    gains = [information_gain(X[:, i], labels) for i in range(X.shape[1])]
    best_feature = np.argmax(gains)

    if feature_names:
        feature_name = feature_names[best_feature]
    else:
        feature_name = f"feature_{best_feature}"

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
    if not isinstance(tree, dict):
        return tree
    feature = list(tree.keys())[0]
    idx = feature_names.index(feature)
    value = sample[idx]
    subtree = tree[feature].get(value, None)
    if subtree is None:
        return None
    return predict_one(subtree, sample, feature_names)


if __name__ == "__main__":
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
    gains_list = [information_gain(X[:, i], labels) for i in range(4)]
    print(f"  → 选择信息增益最大的特征 '{names[np.argmax(gains_list)]}' 作为根节点")

    print("\n预测测试:")
    test = np.array([0, 1, 0, 0])
    print(f"  输入: 天气=晴, 温度=适中, 湿度=高, 风力=弱 → 预测: {predict_one(tree, test, names)}")
