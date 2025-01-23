import hdbscan
import numpy as np


class HdbscanCluster():
    def __init__(self):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,          # 每个簇至少包含 5 个点
            min_samples=20,               # 每个核心点至少需要 1 个邻居
            cluster_selection_epsilon=0.1, # 点之间的距离不超过 2 的区域才会被聚类
            metric='euclidean'           # 使用欧几里得距离
        )
    def predict(self, points):
        # # 准备数据
        # points = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])  # 替换为实际数据
        # 进行聚类
        clusters = self.clusterer.fit_predict(points)

        # 找到每个簇的中心点
        unique_clusters = np.unique(clusters)
        centers = []

        for cluster in unique_clusters:
            if cluster != -1:  # -1 表示噪声
                cluster_points = points[clusters == cluster]
                center = np.mean(cluster_points, axis=0)
                center = np.round(center).astype(int) 
                centers.append(center)

        # 输出结果
        # print("聚类中心点：", centers)
        return centers


