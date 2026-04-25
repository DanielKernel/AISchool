"""
决策树与深度学习基础算法实现
涵盖：决策树核心（信息增益）、卷积（滑动窗口）、Max Pooling（最大池化）
每个算法仅用 numpy，不依赖深度学习框架，适合考试手写理解。
"""

import numpy as np


# ============================================================
# 1. 决策树——信息增益
# ============================================================
# 原理：
#   信息增益 = 父节点的熵 - 按特征分裂后各子节点的加权熵之和
#   选信息增益最大的特征作为最优分裂点
# 关键公式：
#   H(S) = -Σ p_i * log2(p_i)
#   Gain(S, A) = H(S) - Σ [|Sv|/|S| * H(Sv)]
# ============================================================

def entropy(y):
    """
    计算标签序列 y 的信息熵

    参数：y : array-like，类别标签

    返回：float，熵值（越小越纯）
    """
    if len(y) == 0:
        return 0.0
    n = len(y)
    # 统计每个类别的比例
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    # 计算熵，规定 0 * log2(0) = 0
    return -np.sum(probs * np.log2(probs + 1e-12))


def information_gain(X_col, y):
    """
    计算某个离散特征列对标签 y 的信息增益

    参数：
        X_col : 1D array，某特征列的所有取值
        y     : 1D array，对应的标签

    返回：
        float，信息增益值（越大该特征越重要）
    """
    parent_entropy = entropy(y)
    n = len(y)

    # 计算按该特征分裂后各子集的加权熵
    feature_values = np.unique(X_col)
    weighted_child_entropy = 0.0
    for val in feature_values:
        mask = X_col == val
        child_y = y[mask]
        weight = len(child_y) / n
        weighted_child_entropy += weight * entropy(child_y)

    return parent_entropy - weighted_child_entropy


def best_split_feature(X, y):
    """
    在所有特征中找信息增益最大的特征

    参数：
        X : 特征矩阵，shape (n_samples, n_features)
        y : 标签，shape (n_samples,)

    返回：
        best_feat_idx : 最优特征的列索引
        best_gain     : 对应的信息增益
    """
    n_features = X.shape[1]
    gains = [information_gain(X[:, j], y) for j in range(n_features)]
    best_feat_idx = int(np.argmax(gains))
    return best_feat_idx, gains[best_feat_idx]


def demo_decision_tree():
    print("=" * 50)
    print("【1. 决策树信息增益演示】")
    # 经典"是否打球"数据集（简化）
    # 特征: [天气(0晴/1阴/2雨), 风力(0弱/1强)]
    # 标签: 0不打球, 1打球
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [2, 0], [2, 1], [0, 0], [2, 0],
        [1, 0], [0, 1],
    ])
    y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])

    print(f"  数据集大小: {len(y)} 条样本")
    print(f"  父节点熵 H(S) = {entropy(y):.4f}")

    gain_weather = information_gain(X[:, 0], y)
    gain_wind    = information_gain(X[:, 1], y)
    print(f"  特征 [天气] 信息增益 = {gain_weather:.4f}")
    print(f"  特征 [风力] 信息增益 = {gain_wind:.4f}")

    best_idx, best_gain = best_split_feature(X, y)
    feat_names = ["天气", "风力"]
    print(f"  最优分裂特征: [{feat_names[best_idx]}]，增益 = {best_gain:.4f}")


# ============================================================
# 2. 卷积（2D 滑动窗口卷积）
# ============================================================
# 原理：
#   卷积核在输入矩阵上逐步滑动（步长 stride）
#   每次对重叠区域做逐元素乘法并求和，得到输出特征图的一个值
# 输出尺寸公式：out = (N - K + 2*P) / S + 1
# ============================================================

def conv2d(input_map, kernel, stride=1, padding=0):
    """
    2D 卷积（单通道，无批量维度）

    参数：
        input_map : 2D numpy array，输入特征图 (H, W)
        kernel    : 2D numpy array，卷积核 (kH, kW)
        stride    : 步长
        padding   : 边缘补零的像素数

    返回：
        output : 2D numpy array，输出特征图
    """
    H, W = input_map.shape
    kH, kW = kernel.shape

    # 边缘补零
    if padding > 0:
        input_map = np.pad(input_map, padding, mode='constant', constant_values=0)
        H_pad, W_pad = input_map.shape
    else:
        H_pad, W_pad = H, W

    # 计算输出尺寸
    out_H = (H_pad - kH) // stride + 1
    out_W = (W_pad - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    # 滑动窗口
    for i in range(out_H):
        for j in range(out_W):
            # 截取输入的对应区域
            region = input_map[i*stride : i*stride+kH,
                               j*stride : j*stride+kW]
            # 逐元素相乘再求和（核心操作）
            output[i, j] = np.sum(region * kernel)

    return output


def demo_conv2d():
    print("=" * 50)
    print("【2. 卷积（滑动窗口）演示】")
    # 4×4 输入
    input_map = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1],
    ], dtype=float)

    # 3×3 边缘检测核（Sobel 水平方向简化版）
    kernel = np.array([
        [1,  0, -1],
        [2,  0, -2],
        [1,  0, -1],
    ], dtype=float)

    output = conv2d(input_map, kernel, stride=1, padding=0)
    print(f"  输入形状: {input_map.shape}")
    print(f"  卷积核形状: {kernel.shape}，步长=1，padding=0")
    print(f"  输出形状: {output.shape}  （公式：(4-3)/1+1 = 2）")
    print(f"  输出:\n{output}")

    # 演示 padding=1 保持输出尺寸与输入相同
    output_padded = conv2d(input_map, kernel, stride=1, padding=1)
    print(f"\n  padding=1 后输出形状: {output_padded.shape}  （公式：(4-3+2)/1+1 = 4）")


# ============================================================
# 3. Max Pooling（最大池化）
# ============================================================
# 原理：
#   在输入矩阵上用固定大小的池化窗口滑动（步长通常等于窗口大小）
#   每次取窗口内的最大值，达到降采样效果
# 与卷积的区别：无可学习参数，只选最大值
# ============================================================

def max_pooling2d(input_map, pool_size=2, stride=None):
    """
    2D Max Pooling（最大池化）

    参数：
        input_map : 2D numpy array，输入特征图 (H, W)
        pool_size : 池化窗口大小（正方形）
        stride    : 步长，默认等于 pool_size（非重叠池化）

    返回：
        output    : 2D numpy array，池化后的特征图
        max_mask  : 记录每个窗口中最大值位置的掩码（反向传播用）
    """
    if stride is None:
        stride = pool_size  # 默认不重叠

    H, W = input_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # 截取池化窗口区域
            region = input_map[i*stride : i*stride+pool_size,
                               j*stride : j*stride+pool_size]
            # 取最大值（核心操作）
            output[i, j] = np.max(region)

    return output


def avg_pooling2d(input_map, pool_size=2, stride=None):
    """
    2D Average Pooling（平均池化，对比 Max Pooling 用）
    """
    if stride is None:
        stride = pool_size

    H, W = input_map.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = input_map[i*stride : i*stride+pool_size,
                               j*stride : j*stride+pool_size]
            output[i, j] = np.mean(region)  # 取均值

    return output


def demo_pooling():
    print("=" * 50)
    print("【3. Max Pooling 演示】")
    input_map = np.array([
        [1,  3,  2,  4],
        [5,  6,  7,  8],
        [3,  2,  1,  0],
        [1,  2,  3,  4],
    ], dtype=float)

    print(f"  输入 (4×4):\n{input_map}")

    # Max Pooling，窗口=2，步长=2
    mp_out = max_pooling2d(input_map, pool_size=2, stride=2)
    print(f"\n  Max Pooling（窗口=2, 步长=2）输出形状: {mp_out.shape}")
    print(f"  输出:\n{mp_out}")
    print(f"  （左上2×2窗口最大值={int(max(1,3,5,6))}，右上={int(max(2,4,7,8))}，...）")

    # Average Pooling 对比
    ap_out = avg_pooling2d(input_map, pool_size=2, stride=2)
    print(f"\n  Average Pooling 对比输出:\n{ap_out}")

    # Max Pooling，步长=1（重叠池化）
    mp_stride1 = max_pooling2d(input_map, pool_size=2, stride=1)
    print(f"\n  Max Pooling（步长=1，重叠）输出形状: {mp_stride1.shape}")
    print(f"  输出:\n{mp_stride1}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("\n==================================================")
    print("   深度学习基础演示（决策树信息增益 / 卷积 / Max Pooling）")
    print("==================================================\n")

    demo_decision_tree()
    print()
    demo_conv2d()
    print()
    demo_pooling()
    print()
    print("所有深度学习基础算法演示完毕！")
