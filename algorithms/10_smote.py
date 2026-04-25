"""
SMOTE 过采样算法
================
核心思想：对少数类样本 → 找其 K 个近邻 → 在样本与近邻之间随机插值生成新样本

口诀：找少数类 → 找 K 近邻 → 随机插值

公式：x_new = x + rand(0,1) * (x_neighbor - x)
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
    synthetic = np.zeros((n_new_samples, n_features))

    for i in range(n_new_samples):
        # Step 1: 随机选一个少数类样本
        idx = np.random.randint(0, n_samples)
        sample = X_minority[idx]

        # Step 2: 找它的 K 个近邻
        distances = np.linalg.norm(X_minority - sample, axis=1)
        # 排除自身 (距离为0的)，取最近的 k 个
        neighbor_indices = np.argsort(distances)[1:k+1]

        # Step 3: 随机选一个近邻，在两者之间插值
        nn_idx = neighbor_indices[np.random.randint(0, len(neighbor_indices))]
        neighbor = X_minority[nn_idx]

        # x_new = x + rand * (neighbor - x)
        lam = np.random.random()
        synthetic[i] = sample + lam * (neighbor - sample)

    return synthetic


# ========== 运行示例 ==========
if __name__ == "__main__":
    np.random.seed(42)

    # 模拟不平衡数据: 多数类 100 个, 少数类 10 个
    X_majority = np.random.randn(100, 2) + [5, 5]
    X_minority = np.random.randn(10, 2) + [0, 0]

    print(f"原始少数类样本数: {len(X_minority)}")
    print(f"原始多数类样本数: {len(X_majority)}")

    # 用 SMOTE 生成 90 个新少数类样本，使两类平衡
    new_samples = smote(X_minority, n_new_samples=90, k=5)
    X_minority_balanced = np.vstack([X_minority, new_samples])

    print(f"\nSMOTE 后少数类样本数: {len(X_minority_balanced)}")
    print(f"新生成样本示例 (前3个):")
    for s in new_samples[:3]:
        print(f"  ({s[0]:.2f}, {s[1]:.2f})")
