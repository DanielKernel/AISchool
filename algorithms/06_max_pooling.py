"""
最大池化 (Max Pooling)
=====================

【算法背景】
池化 (Pooling) 是 CNN 中的下采样操作，通常紧跟在卷积层之后。
作用:
  1. 减小特征图尺寸 → 减少计算量和参数数量
  2. 增大感受野 → 每个输出神经元"看到"更大范围的输入
  3. 提供平移不变性 → 图像小幅移动不影响输出

Max Pooling 是最常用的池化方式：取窗口内的最大值。
直觉：保留窗口内最强的特征响应。

【适用场景】
- CNN 中的下采样层（几乎所有 CNN 架构都会用到）
- 特征降维

【算法流程】
    ┌──────────────────────────────┐
    │ Step 1: 计算输出尺寸           │
    │  out = (H - pool_size)/s + 1  │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: 双层循环滑动窗口       │
    │  for i in range(out_H):      │
    │    for j in range(out_W):    │
    │      region = 提取窗口       │
    │      output[i,j] =          │
    │        max(region)           │
    └──────────────────────────────┘

【核心公式】
  输出尺寸: out = (input_size - pool_size) / stride + 1
  池化运算: output[i,j] = max(image[i*s : i*s+p, j*s : j*s+p])

【口诀】双层循环 → 提取窗口 → 取最大值

【记忆要点 - 考试必背】
  1. 输出尺寸公式和卷积一样（没有 padding 版本）
  2. 默认 stride = pool_size（窗口不重叠）
  3. 核心只有一行: output[i,j] = np.max(region)
  4. 与卷积的代码框架完全一样，只是把 np.sum(region * kernel) 换成 np.max(region)

【Max Pooling vs Average Pooling（常考）】
  Max Pooling:     取窗口最大值 → 保留最强特征 → 更常用
  Average Pooling: 取窗口平均值 → 保留整体信息 → 常用于全局池化

【具体例子（手算必会）】
  输入 4x4:                 2x2 Max Pooling (stride=2):
  ┌───┬───┬───┬───┐        ┌───┬───┐
  │ 1 │ 3 │ 2 │ 4 │        │ 6 │ 4 │   max(1,3,5,6)=6, max(2,4,1,2)=4
  ├───┼───┼───┼───┤   →    ├───┼───┤
  │ 5 │ 6 │ 1 │ 2 │        │ 9 │ 8 │   max(7,2,4,9)=9, max(3,8,1,5)=8
  ├───┼───┼───┼───┤        └───┴───┘
  │ 7 │ 2 │ 3 │ 8 │
  ├───┼───┼───┼───┤
  │ 4 │ 9 │ 1 │ 5 │
  └───┴───┴───┴───┘

【易错点】
  - stride 通常等于 pool_size（不重叠），但不是必须的
  - 池化没有可学习参数！（和卷积不同，卷积核是需要学习的）
  - 池化不做零填充（通常情况下）

【复杂度】
  时间: O(out_H * out_W * p²)   遍历每个窗口取最大值
  空间: O(out_H * out_W)        输出矩阵
"""
import numpy as np


def max_pooling(image, pool_size=2, stride=2):
    """
    参数:
        image:     输入矩阵, shape (H, W)
        pool_size: 池化窗口大小
        stride:    步长 (通常等于 pool_size)
    返回:
        output:    池化结果
    """
    H, W = image.shape

    # 计算输出尺寸
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))

    # 滑动窗口取最大值
    for i in range(out_H):
        for j in range(out_W):
            region = image[i*stride : i*stride+pool_size,
                           j*stride : j*stride+pool_size]
            output[i, j] = np.max(region)

    return output


def avg_pooling(image, pool_size=2, stride=2):
    """平均池化 — 作为对比学习"""
    H, W = image.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = image[i*stride : i*stride+pool_size,
                           j*stride : j*stride+pool_size]
            output[i, j] = np.mean(region)

    return output


# ========== 运行示例 ==========
if __name__ == "__main__":
    image = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [7, 2, 3, 8],
        [4, 9, 1, 5],
    ], dtype=float)

    max_result = max_pooling(image, pool_size=2, stride=2)
    avg_result = avg_pooling(image, pool_size=2, stride=2)

    print("Max Pooling 示例:")
    print("=" * 40)
    print("输入 (4x4):")
    print(image)
    print(f"\n2x2 Max Pooling 结果 ({max_result.shape[0]}x{max_result.shape[1]}):")
    print(max_result)

    print()
    print("--- 手动验证 ---")
    print(f"  左上角 2x2: [{image[0,0]}, {image[0,1]}, {image[1,0]}, {image[1,1]}] → max = {max_result[0,0]:.0f}")
    print(f"  右上角 2x2: [{image[0,2]}, {image[0,3]}, {image[1,2]}, {image[1,3]}] → max = {max_result[0,1]:.0f}")
    print(f"  左下角 2x2: [{image[2,0]}, {image[2,1]}, {image[3,0]}, {image[3,1]}] → max = {max_result[1,0]:.0f}")
    print(f"  右下角 2x2: [{image[2,2]}, {image[2,3]}, {image[3,2]}, {image[3,3]}] → max = {max_result[1,1]:.0f}")

    print(f"\n2x2 Average Pooling 对比:")
    print(avg_result)
