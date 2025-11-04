import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import copy
import numpy as np
# 导入我们之前定义的特征分布工具类
from feature_distribution import FeatureDistribution


class PfedfdaClient:
    """
    pFedFDA 客户端实现。每个客户端维护本地特征提取器 f_i 和本地分类器 g_i。
    """

    def __init__(self, client_id, data, feature_extractor, classifier, device,
                 lr=1e-4, weight_decay=1e-5, feature_dim=None, gmm_components=5, reg_lambda=0.01):
        """
        Args:
            client_id (...): 客户端ID。
            data (...): 客户端本地数据。
            feature_extractor (nn.Module): 特征提取器 f (GNN)。
            classifier (nn.Module): 本地分类器 g (ResMLP)。
            device (torch.device): 设备。
            feature_dim (int): 特征维度 (f 的输出维度)。
            gmm_components (int): GMM 分量数。
            reg_lambda (float): 分布正则化损失 L_Reg 的权重。
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # 本地模型：特征提取器 f_i 和分类器 g_i
        self.feature_extractor = feature_extractor.to(device)  # f_i
        self.classifier = classifier.to(device)  # g_i

        # 特征分布参数
        self.feature_dim = feature_dim
        self.gmm_components = gmm_components
        self.reg_lambda = reg_lambda

        # 本地特征分布 P_i
        self.local_distribution = FeatureDistribution(feature_dim, gmm_components, device)
        # 全局特征分布 P_G 的参数（每次通信轮次由 Server 下发）
        self.global_distribution_params = None

        self.lr = lr
        self.weight_decay = weight_decay

    def get_optimizer(self):
        """为 f_i 和 g_i 创建优化器。"""
        # 注意：pFedFDA 客户端同时训练 f_i 和 g_i
        params = list(self.feature_extractor.parameters()) + \
                 list(self.classifier.parameters())

        return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

    # --------------------------------------------------------------------------
    # 【核心】本地特征分布估计
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def estimate_local_distribution(self):
        """
        使用当前本地的特征提取器 f_i 提取特征，并更新本地 GMM P_i。
        """
        self.feature_extractor.eval()
        # 提取所有训练节点的特征
        z_features = self.feature_extractor(self.data.x, self.data.edge_index)

        # 仅使用有标签的训练节点特征来拟合分布 (确保与监督训练数据一致)
        train_mask = self.data.train_mask & (self.data.y >= 0)
        local_features = z_features[train_mask]

        # 拟合本地 GMM
        self.local_distribution.fit_distribution(local_features)

    # --------------------------------------------------------------------------
    # 【核心】设置参数：Server 下发共享 f_G 和全局分布 P_G 的参数
    # --------------------------------------------------------------------------
    def set_parameters(self, parameters):
        """
        设置共享的特征提取器 f_G 和全局分布 P_G 的参数。

        Args:
            parameters (Dict): 包含 f_G, g_G (用于本地初始化) 和 P_G 参数。
        """
        # 1. 加载共享特征提取器 f_G 到 f_i
        self.feature_extractor.load_state_dict(parameters["feature_extractor"])

        # 2. 加载全局分类器 g_G 到本地分类器 g_i (作为初始化)
        self.classifier.load_state_dict(parameters["classifier_global"])

        # 3. 存储全局特征分布 P_G 的参数
        self.global_distribution_params = parameters["global_distribution"]

        # 4. 在新一轮开始时，立即更新本地特征分布 P_i
        self.estimate_local_distribution()

    # --------------------------------------------------------------------------
    # 【核心】计算特征分布正则化损失 L_Reg
    # --------------------------------------------------------------------------
    def _compute_regularization_loss(self, current_features: torch.Tensor):
        """
        计算特征分布正则化损失 L_Reg = - E_{x ~ P_i} [log P_G(x)]。
        这近似于 KL(P_i || P_G)。

        Args:
            current_features (torch.Tensor): 当前批次/轮次 f_i 提取的特征 (N, D)。

        Returns:
            torch.Tensor: 正则化损失 L_Reg。
        """
        if self.global_distribution_params is None or current_features.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # 1. 创建一个临时的 P_G 实例来计算 log_pdf
        pg_temp = FeatureDistribution(self.feature_dim, self.gmm_components, self.device)
        pg_temp.weights = self.global_distribution_params["weights"]
        pg_temp.means = self.global_distribution_params["means"]
        pg_temp.covariances = self.global_distribution_params["covariances"]
        pg_temp.num_components = pg_temp.weights.shape[0]  # 确保组件数正确

        # 2. 计算当前特征在 P_G 下的对数概率密度
        # log_prob 形状: (N,)
        log_prob_pg = pg_temp.log_pdf(current_features)

        # 3. 计算期望 (E_{x ~ P_i} [log P_G(x)])
        # 由于我们使用训练集所有节点作为 "batch size"，直接计算平均值
        expected_log_prob = log_prob_pg.mean()

        # 4. L_Reg = - 期望 (负号是为了最大化 log P_G，即最小化 KL 散度)
        loss_reg = -expected_log_prob

        return loss_reg

    # --------------------------------------------------------------------------
    # 【核心】本地训练循环
    # --------------------------------------------------------------------------
    def local_train(self, epochs):

        optimizer = self.get_optimizer()

        self.feature_extractor.train()
        self.classifier.train()

        train_mask = self.data.train_mask
        valid_labels_mask = (self.data.y >= 0)
        final_train_mask = train_mask & valid_labels_mask

        if final_train_mask.sum().item() == 0:
            return 0.0

        for _ in range(epochs):
            optimizer.zero_grad()

            # 1. 前向传播
            z_features = self.feature_extractor(self.data.x, self.data.edge_index)
            logits = self.classifier(z_features)

            # 2. L_Loc (本地监督损失)
            train_logits = logits[final_train_mask]
            train_labels = self.data.y[final_train_mask]
            loss_loc = self.criterion(train_logits, train_labels)

            # 3. L_Reg (特征分布正则化损失)
            # 使用训练集特征来计算正则化
            train_features = z_features[final_train_mask]
            loss_reg = self._compute_regularization_loss(train_features)

            # 4. 总损失: L_Total = L_Loc + lambda * L_Reg
            loss = loss_loc + self.reg_lambda * loss_reg

            loss.backward()
            optimizer.step()

        # 训练结束后，更新本地分布 P_i，准备下一次正则化计算
        self.estimate_local_distribution()

        return loss.item()

    # --------------------------------------------------------------------------
    # 上传：只返回训练后的特征提取器 f_i 和本地分布 P_i 的参数
    # --------------------------------------------------------------------------
    def get_trained_update(self):
        """
        返回本地训练后的模型参数和特征分布参数。
        """
        # f_i 用于 Server FedAvg
        f_update = copy.deepcopy(self.feature_extractor.state_dict())

        # P_i 参数用于 Server 更新 P_G
        pi_params = {
            "weights": self.local_distribution.weights.clone().detach(),
            "means": self.local_distribution.means.clone().detach(),
            "covariances": self.local_distribution.covariances.clone().detach(),
        }

        return {
            "feature_extractor": f_update,
            "local_distribution": pi_params,
            "num_nodes": self.data.train_mask.sum().item()  # 用于加权聚合
        }

    # --------------------------------------------------------------------------
    # 评估：使用本地模型 (f_i, g_i) 评估
    # --------------------------------------------------------------------------
    def evaluate(self, use_test=False):
        """
        评估当前客户端的个性化模型 (f_i, g_i)。
        """
        self.feature_extractor.eval()
        self.classifier.eval()

        with torch.no_grad():
            z_features = self.feature_extractor(self.data.x, self.data.edge_index)
            logits = self.classifier(z_features)

            if use_test:
                initial_mask = self.data.test_mask
            else:
                initial_mask = self.data.val_mask

            valid_labels_mask = (self.data.y >= 0)
            final_eval_mask = initial_mask & valid_labels_mask

            if final_eval_mask.sum().item() == 0:
                return 0.0, 0.0, 0.0, 0.0

            eval_logits = logits[final_eval_mask]
            y_true = self.data.y[final_eval_mask]

            y_pred_labels = eval_logits.argmax(dim=1)

            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred_labels.cpu().numpy()

            acc = accuracy_score(y_true_np, y_pred_np)
            recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)

        return acc, recall, precision, f1
