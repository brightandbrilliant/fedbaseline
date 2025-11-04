import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
import numpy as np
from feature_distribution import FeatureDistribution  # 导入特征分布工具


class PfedfdaServer:
    """
    pFedFDA 服务器实现。管理共享 GNN (f)、全局分类器 (g_G) 和全局特征分布 (P_G)。
    """

    def __init__(self, feature_extractor, classifier_global, device, feature_dim,
                 num_classes, gmm_components=5,
                 gen_lr=0.005, gen_epochs=1):
        """
        初始化 pFedFDA 服务器端。

        Args:
            feature_extractor (nn.Module): 共享特征提取器 f。
            classifier_global (nn.Module): 全局生成式分类器 g_G。
            device (torch.device): 计算设备。
            feature_dim (int): 特征维度 (f 的输出维度)。
            num_classes (int): 类别数。
            gmm_components (int): GMM 分量数。
            gen_lr (float): 全局分类器生成式训练的学习率。
            gen_epochs (int): 全局分类器生成式训练的轮数。
        """
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.gmm_components = gmm_components

        # 共享模型
        self.feature_extractor = feature_extractor.to(device)  # f
        # 全局生成式分类器
        self.classifier_global = classifier_global.to(device)  # g_G

        # 全局特征分布 P_G
        self.global_distribution = FeatureDistribution(feature_dim, gmm_components, device)

        # 生成式训练参数
        self.gen_lr = gen_lr
        self.gen_epochs = gen_epochs
        # 辅助 Server 端训练的分类器损失，例如 KL 散度或交叉熵
        self.criterion_gen = nn.CrossEntropyLoss()

        # --------------------------------------------------------------------------

    # 【核心 1】获取全局参数：f, g_G, P_G
    # --------------------------------------------------------------------------
    def get_global_parameters(self) -> Dict[str, Any]:
        """
        获取当前全局模型 f、g_G 和 P_G 的参数。
        """
        pg_params = {
            "weights": self.global_distribution.weights.clone().detach(),
            "means": self.global_distribution.means.clone().detach(),
            "covariances": self.global_distribution.covariances.clone().detach(),
        }

        return {
            # 共享特征提取器
            "feature_extractor": copy.deepcopy(self.feature_extractor.state_dict()),
            # 全局分类器 (作为客户端本地 g_i 的初始值)
            "classifier_global": copy.deepcopy(self.classifier_global.state_dict()),
            # 全局特征分布 P_G 参数
            "global_distribution": pg_params
        }

    # --------------------------------------------------------------------------
    # 【核心 2】设置全局参数 (用于加载最佳模型)
    # --------------------------------------------------------------------------
    def set_global_parameters(self, parameters: Dict[str, Any]):
        """
        设置全局模型的参数。
        """
        # 1. 共享特征提取器
        self.feature_extractor.load_state_dict(parameters["feature_extractor"])

        # 2. 全局分类器
        self.classifier_global.load_state_dict(parameters["classifier_global"])

        # 3. 全局特征分布
        pg_params = parameters["global_distribution"]
        self.global_distribution.weights = pg_params["weights"].to(self.device)
        self.global_distribution.means = pg_params["means"].to(self.device)
        self.global_distribution.covariances = pg_params["covariances"].to(self.device)
        self.global_distribution.num_components = self.global_distribution.weights.shape[0]

    # --------------------------------------------------------------------------
    # 【核心 3】聚合特征提取器 f (FedAvg)
    # --------------------------------------------------------------------------
    def aggregate_feature_extractor(self, client_updates: List[Dict[str, Any]]):
        """
        对客户端上传的 f_i 进行 FedAvg 聚合。

        Args:
            client_updates (List[Dict]): 客户端上传的结果列表。
        """
        if not client_updates:
            return

        total_samples = sum(update["num_nodes"] for update in client_updates)

        global_state = self.feature_extractor.state_dict()
        new_state = copy.deepcopy(global_state)

        for key in new_state.keys():
            # 计算加权平均： sum(w_i * state_i)
            new_state[key] = sum(
                (update["feature_extractor"][key] * (update["num_nodes"] / total_samples))
                for update in client_updates
            )

        self.feature_extractor.load_state_dict(new_state)

    # --------------------------------------------------------------------------
    # 【核心 4】更新全局特征分布 P_G
    # --------------------------------------------------------------------------
    def update_global_distribution(self, client_updates: List[Dict[str, Any]]):
        """
        使用客户端上传的本地特征分布 P_i (参数) 更新全局分布 P_G。
        简单方法：使用所有客户端的均值和协方差进行加权平均。
        """
        if not client_updates:
            return

        total_samples = sum(update["num_nodes"] for update in client_updates)

        # 收集所有客户端的 GMM 参数
        all_weights = []
        all_means = []
        all_covariances = []
        all_nodes = []

        for update in client_updates:
            params = update["local_distribution"]
            all_weights.append(params["weights"])
            all_means.append(params["means"])
            all_covariances.append(params["covariances"])
            all_nodes.append(update["num_nodes"])

        # 为了简化聚合，我们假设所有 P_i 拥有相同的组件数 self.gmm_components。
        # 如果组件数不同，需要更复杂的合并 GMM 逻辑。

        # 1. 聚合权重 (weights)
        # pi_G = sum_{i} (N_i/N_total) * pi_i
        # GMM 聚合在 IFCA 论文中有讨论（虽然不是 pFedFDA），这里采用最简单加权平均
        agg_weights = torch.zeros(self.gmm_components, device=self.device)

        for pi_i, N_i in zip(all_weights, all_nodes):
            # 确保客户端上传的组件数匹配
            if pi_i.shape[0] == self.gmm_components:
                agg_weights += pi_i * (N_i / total_samples)

        # 2. 聚合均值和协方差 (这里仅做简单的加权平均，实际 GMM 合并更复杂)
        agg_means = torch.zeros(self.gmm_components, self.feature_dim, device=self.device)
        agg_covs = torch.zeros(self.gmm_components, self.feature_dim, self.feature_dim, device=self.device)

        for mu_i, sigma_i, N_i in zip(all_means, all_covariances, all_nodes):
            if mu_i.shape[0] == self.gmm_components:
                agg_means += mu_i * (N_i / total_samples)
                agg_covs += sigma_i * (N_i / total_samples)

        # 更新全局分布
        self.global_distribution.weights = agg_weights
        self.global_distribution.means = agg_means
        self.global_distribution.covariances = agg_covs
        self.global_distribution.num_components = self.gmm_components

        print(f"Global distribution P_G updated with {len(client_updates)} clients.")

    # --------------------------------------------------------------------------
    # 【核心 5】训练全局生成式分类器 g_G (Optimization Step)
    # --------------------------------------------------------------------------
    def train_global_classifier(self, num_samples=1000):
        """
        在 Server 端使用全局特征分布 P_G 进行生成式训练。

        Args:
            num_samples (int): 从 P_G 中采样的特征点数量。
        """
        # 只有 g_G 参与训练
        self.classifier_global.train()
        # f 保持冻结 (使用 FedAvg更新后的 f)
        self.feature_extractor.eval()

        optimizer = optim.Adam(self.classifier_global.parameters(), lr=self.gen_lr)

        print(f"Training g_G using P_G with {num_samples} samples over {self.gen_epochs} epochs.")

        for epoch in range(self.gen_epochs):
            optimizer.zero_grad()

            # 1. 从 P_G 中采样特征 z (形状: N, D)
            # 注意: P_G 拟合自客户端特征，通常是无标签的
            # 然而，L_Gen 需要标签。论文假设特征空间的邻域具有相似的标签，
            # 或者使用全局标签信息。由于我们没有全局标签，这里简化处理，
            # 暂时假设我们可以访问一个与特征分布相关的伪标签机制。
            #
            # 实际操作中，pFedFDA 假设全局分类器 g_G 是一个多任务学习的头部，
            # 或者使用一种机制，使得采样特征可以得到伪标签。
            #
            # **简化处理:** 我们使用全局分类器 g_G 对采样的特征进行预测，
            # 然后使用一种自监督/一致性损失，或在多任务设置下使用全局共享标签。
            #
            # 遵循论文精神，L_Gen 目标是**最小化全局特征空间上的预测方差/不确定性**。

            # **此处采用 L_Gen 变体:** 最小化预测的熵 (最大化置信度)
            # 这鼓励 g_G 在全局特征空间上做出高置信度的预测。

            sampled_features = self.global_distribution.sample(num_samples)

            # 2. 前向传播
            logits = self.classifier_global(sampled_features)
            probs = torch.softmax(logits, dim=1)

            # 3. 计算 L_Gen: 最小化预测熵 (Entropy Minimization)
            # H(p) = - sum(p * log(p))
            log_probs = torch.log(probs.clamp(min=1e-8))
            entropy = - (probs * log_probs).sum(dim=1).mean()

            # L_Gen = 熵最小化
            loss_gen = entropy

            loss_gen.backward()
            optimizer.step()

        print(f"g_G training finished. Final L_Gen (Entropy): {loss_gen.item():.4f}")

    def distribute_parameters(self, clients: List[Any]):
        """
        将所有全局参数 (f, g_G, P_G) 下发给所有客户端。
        """
        params = self.get_global_parameters()
        for client in clients:
            # client.set_parameters 接收 f, g_G 和 P_G
            client.set_parameters(params)
            