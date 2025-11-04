import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from sklearn.mixture import GaussianMixture  # 使用 scikit-learn 来进行 GMM 估计


class FeatureDistribution:
    """
    负责特征分布 (P_G 或 P_i) 的估计、维护和操作的类。
    在 pFedFDA 中，我们使用高斯混合模型 (GMM) 来近似特征分布。
    """

    def __init__(self, feature_dim: int, num_components: int, device: torch.device):
        """
        初始化特征分布对象。
        Args:
            feature_dim (int): 特征向量的维度 (即 GNN 的 output_dim)。
            num_components (int): GMM 中的高斯分量数量。
            device (torch.device): 计算设备。
        """
        self.feature_dim = feature_dim
        self.num_components = num_components
        self.device = device

        # GMM 参数：权重 (pi_k)、均值 (mu_k)、协方差 (Sigma_k)
        # 初始值通常是随机的或在第一次拟合后确定
        self.weights = torch.ones(num_components, device=device) / num_components
        self.means = torch.randn(num_components, feature_dim, device=device)
        self.covariances = torch.stack([torch.eye(feature_dim, device=device) for _ in range(num_components)])

    def fit_distribution(self, features: torch.Tensor, max_iter: int = 100):
        """
        使用本地特征数据拟合 GMM 参数。

        Args:
            features (torch.Tensor): 形状为 (N, feature_dim) 的特征向量。
            max_iter (int): GMM 拟合的最大迭代次数。
        """
        if features.numel() == 0:
            print("警告: 尝试使用空特征拟合分布。分布参数保持不变。")
            return

        # 1. 移到 CPU，使用 sklearn.mixture.GaussianMixture 进行估计
        features_np = features.detach().cpu().numpy()

        # 确保数据量足够大，否则降低分量数
        n_samples = features_np.shape[0]
        n_components_fit = min(self.num_components, n_samples)

        if n_components_fit == 0:  # 如果没有数据，直接返回
            return

        # 2. 初始化 GMM 模型并拟合
        gmm = GaussianMixture(
            n_components=n_components_fit,
            covariance_type='full',
            max_iter=max_iter,
            tol=1e-3,
            n_init=1,  # 简化设置
            init_params='random'
        )

        try:
            gmm.fit(features_np)
        except ValueError as e:
            # 拟合失败（例如协方差矩阵奇异），使用最少的分量或默认值
            print(f"GMM 拟合失败 ({e})，尝试使用 1 个分量。")
            gmm = GaussianMixture(n_components=1, covariance_type='full', max_iter=max_iter, n_init=1)
            gmm.fit(features_np)

        # 3. 将拟合结果转回 PyTorch Tensor 并存储
        self.weights = torch.tensor(gmm.weights_, dtype=torch.float32, device=self.device)
        self.means = torch.tensor(gmm.means_, dtype=torch.float32, device=self.device)
        self.covariances = torch.tensor(gmm.covariances_, dtype=torch.float32, device=self.device)

        # 修正分量数量，如果拟合时减少了
        self.num_components = self.weights.shape[0]

    def log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入特征 x 在当前 GMM 分布下的对数概率密度。
        用于计算 $L_{Reg}$ (KL散度近似) 和 $L_{Gen}$。

        Args:
            x (torch.Tensor): 形状为 (N, feature_dim) 的特征点。

        Returns:
            torch.Tensor: 形状为 (N,) 的对数概率密度。
        """
        # 确保 GMM 参数有效
        if self.num_components == 0:
            return torch.zeros(x.shape[0], device=self.device) - float('inf')

        # 1. 初始化高斯分布分量
        mix_dist = []
        for k in range(self.num_components):
            # 避免协方差矩阵非正定，增加微小扰动
            cov = self.covariances[k] + torch.eye(self.feature_dim, device=self.device) * 1e-6
            mix_dist.append(MultivariateNormal(loc=self.means[k], covariance_matrix=cov))

        log_pdfs_k = []
        for k in range(self.num_components):
            # 计算每个特征点在第 k 个高斯分量下的对数概率密度
            log_pdfs_k.append(mix_dist[k].log_prob(x).unsqueeze(1))

        # 形状: (N, num_components)
        log_pdfs_k = torch.cat(log_pdfs_k, dim=1)

        # 对数权重
        log_weights = torch.log(self.weights.clamp(min=1e-8))  # 避免 log(0)

        # log(sum_k (pi_k * p_k(x))) 使用 LogSumExp 公式：
        # log(sum(exp(log(pi_k) + log(p_k(x)))))
        log_weighted_pdfs = log_pdfs_k + log_weights

        # 返回每个特征点的 GMM 对数概率密度
        return torch.logsumexp(log_weighted_pdfs, dim=1)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        从当前的 GMM 分布中采样特征点。

        Args:
            num_samples (int): 采样的数量。

        Returns:
            torch.Tensor: 形状为 (num_samples, feature_dim) 的采样特征。
        """
        if self.num_components == 0:
            return torch.zeros((num_samples, self.feature_dim), device=self.device)

        # 1. 确定每个分量需要采样的数量
        # Multinomial 分布根据权重 self.weights 随机选择要采样的分量
        component_indices = torch.multinomial(self.weights, num_samples=num_samples, replacement=True)

        # 2. 从每个选定的高斯分量中采样
        sampled_features = torch.zeros(num_samples, self.feature_dim, device=self.device)

        for k in range(self.num_components):
            # 找到所有被分配给分量 k 的索引
            indices_k = (component_indices == k).nonzero(as_tuple=True)[0]
            count_k = len(indices_k)

            if count_k > 0:
                # 避免协方差矩阵非正定
                cov = self.covariances[k] + torch.eye(self.feature_dim, device=self.device) * 1e-6
                dist_k = MultivariateNormal(loc=self.means[k], covariance_matrix=cov)

                # 从分量 k 采样
                samples_k = dist_k.sample(sample_shape=(count_k,))

                # 放入最终的张量中
                sampled_features[indices_k] = samples_k

        return sampled_features
