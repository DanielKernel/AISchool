"""
SMOTE 过采样算法
================

【算法背景】
SMOTE (Synthetic Minority Over-sampling Technique) 由 Chawla 等人于 2002 年提出。
用于解决分类任务中的数据不平衡问题。

问题场景: 假设信用卡欺诈检测中，正常交易 10000 条，欺诈交易仅 100 条。
直接训练分类器会严重偏向多数类（全预测"正常"也能达到 99% 准确率）。

SMOTE 的解决方案: 通过在少数类样本之间"插值"来生成新的合成样本，
而不是简单地复制已有样本（复制会导致过拟合）。

【适用场景】
- 信用卡欺诈检测、疾病诊断、故障检测等正负样本极不平衡的场景

【算法流程】
    ┌──────────────────────────────┐
    │ 输入: 少数类样本集 X_minority  │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ 重复 n_new_samples 次:        │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 1: 随机选一个少数类样本 x  │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 2: 计算 x 到所有少数类    │
    │   样本的距离，找 K 个最近邻    │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │ Step 3: 随机选一个近邻 x_nn    │
    │   生成新样本:                  │
    │   x_new = x + λ*(x_nn - x)   │
    │   λ ∈ [0, 1) 随机              │
    └──────────────────────────────┘

    生成的 x_new 落在 x 和 x_nn 的连线上。

【核心公式 ★★★ 必背】
  x_new = x + λ * (x_neighbor - x),   λ ~ Uniform(0, 1)

  几何含义: 在 x 和 x_neighbor 之间的连线上随机取一点
  - λ=0 时 x_new = x (原点)
  - λ=1 时 x_new = x_neighbor (邻居)
  - 0<λ<1 时在两点之间

【口诀】找少数类 → 找 K 近邻 → 随机插值

【记忆要点 - 考试必背】
  1. 选样本: idx = np.random.randint(0, n_samples)
  2. 找近邻: distances = np.linalg.norm(X - sample, axis=1)
             neighbor_indices = np.argsort(distances)[1:k+1]
     ★ [1:k+1] 而不是 [:k]，因为 [0] 是自身（距离=0）
  3. 插值公式: x_new = sample + λ * (neighbor - sample)
     ★ λ = np.random.random()，范围 [0, 1)
  4. 只对少数类做！多数类不需要

【与其他处理不平衡的方法对比（常考）】
  过采样:
    - 简单复制: 复制少数类样本 → 容易过拟合
    - SMOTE:   合成新样本      → 增加多样性，更好

  欠采样:
    - 随机删除多数类样本 → 丢失信息

  其他:
    - 类别权重调整 (class_weight)
    - 集成方法 (如 EasyEnsemble)

【SMOTE 的变种（了解）】
  Borderline-SMOTE: 只对边界附近的少数类样本做插值
  ADASYN:           根据分布密度自适应生成样本

【易错点】
  - 近邻索引从 [1:] 开始，跳过自身
  - 只在少数类的特征空间内操作，不涉及多数类
  - 生成的新样本只有特征，标签自动和少数类相同
  - K 不能大于少数类样本数 - 1
  - SMOTE 应该在训练集上做，不能用在测试集上！

【复杂度】
  时间: O(n_new * n_minority * d)   每个新样本要算到所有少数类样本的距离
  空间: O(n_new * d)                存储新样本
"""
import numpy as np


def smote(X_minority, n_new_samples, k=5):
    """
    参数:
        X_minority:    少数类样本, shape (n_samples, n_features)
        n_new_samples: 需要生成的新样本数量
        k:             近邻数量
    返回:
        synthetic: 生成的新样本, shape (n_new_samples, n_features)
    """
    n_samples, n_features = X_minority.shape
    k = min(k, n_samples - 1)  # K 不能超过样本数-1
    synthetic = np.zeros((n_new_samples, n_features))

    for i in range(n_new_samples):
        # Step 1: 随机选一个少数类样本
        idx = np.random.randint(0, n_samples)
        sample = X_minority[idx]

        # Step 2: 找它的 K 个近邻
        distances = np.linalg.norm(X_minority - sample, axis=1)
        neighbor_indices = np.argsort(distances)[1:k+1]  # 跳过自身

        # Step 3: 随机选一个近邻，在两者之间插值
        nn_idx = neighbor_indices[np.random.randint(0, len(neighbor_indices))]
        neighbor = X_minority[nn_idx]

        # ★ 核心公式: x_new = x + λ * (neighbor - x)
        lam = np.random.random()
        synthetic[i] = sample + lam * (neighbor - sample)

    return synthetic


# ========== 运行示例 ==========
if __name__ == "__main__":
    np.random.seed(42)

    # 模拟不平衡数据
    X_majority = np.random.randn(100, 2) + [5, 5]
    X_minority = np.random.randn(10, 2) + [0, 0]

    print("SMOTE 过采样:")
    print("=" * 40)
    print(f"原始少数类样本数: {len(X_minority)}")
    print(f"原始多数类样本数: {len(X_majority)}")
    print(f"不平衡比例: 1:{len(X_majority)//len(X_minority)}")

    new_samples = smote(X_minority, n_new_samples=90, k=5)
    X_minority_balanced = np.vstack([X_minority, new_samples])

    print(f"\nSMOTE 后少数类样本数: {len(X_minority_balanced)}")
    print(f"平衡后比例: 1:{len(X_majority)//len(X_minority_balanced)}")

    print(f"\n新生成样本示例 (前3个):")
    for s in new_samples[:3]:
        print(f"  ({s[0]:.2f}, {s[1]:.2f})")

    print()
    print("--- 理解辅助：一次插值过程追踪 ---")
    np.random.seed(0)
    idx = 0
    sample = X_minority[idx]
    distances = np.linalg.norm(X_minority - sample, axis=1)
    nn_indices = np.argsort(distances)[1:6]
    nn_idx = nn_indices[0]
    neighbor = X_minority[nn_idx]
    lam = 0.4
    x_new = sample + lam * (neighbor - sample)
    print(f"  选中样本 x = ({sample[0]:.2f}, {sample[1]:.2f})")
    print(f"  最近邻 x_nn = ({neighbor[0]:.2f}, {neighbor[1]:.2f})")
    print(f"  λ = {lam}")
    print(f"  x_new = x + {lam} * (x_nn - x) = ({x_new[0]:.2f}, {x_new[1]:.2f})")
    print(f"  → 新样本落在 x 和 x_nn 的连线上，距 x 约 {lam*100:.0f}% 处")
