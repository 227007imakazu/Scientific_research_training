import numpy as np


def k_means(data, k, max_iters=100):
    # 随机初始化k个中心点
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for iter in range(max_iters):
        # 存储旧的中心点用于收敛判断
        old_centroids = centroids.copy()

        # 计算每个点到各个中心点的距离，并分配到最近的中心点
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # 打印中间迭代结果
        print(f"迭代 {iter + 1}:")
        print("中心点:")
        for i, centroid in enumerate(centroids):
            print(f"类别{i}: ({centroid[0]:.4f}, {centroid[1]:.4f})")
        print()

        # 更新中心点
        for i in range(k):
            if np.sum(labels == i) > 0:
                centroids[i] = np.mean(data[labels == i], axis=0)

        # 判断是否收敛
        if np.all(np.abs(old_centroids - centroids) < 1e-4):
            break

    return labels, centroids


# 从文件加载数据
data = np.loadtxt('hw6_data.txt')

# K=2的聚类
print("K=2的聚类结果：")
labels2, centroids2 = k_means(data, 2)
print("最终聚类结果：")
print("中心点:")
for i, centroid in enumerate(centroids2):
    print(f"类别{i}: ({centroid[0]:.4f}, {centroid[1]:.4f})")
print("各点的类别标签:", labels2)
print("\n" + "=" * 50 + "\n")

# K=3的聚类
print("K=3的聚类结果：")
labels3, centroids3 = k_means(data, 3)
print("最终聚类结果：")
print("中心点:")
for i, centroid in enumerate(centroids3):
    print(f"类别{i}: ({centroid[0]:.4f}, {centroid[1]:.4f})")
print("各点的类别标签:", labels3)