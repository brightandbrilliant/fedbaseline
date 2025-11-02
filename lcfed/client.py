import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import copy
from typing import Dict, Any


class LcClient:
    """
    LCFed 客户端实现。客户端的模型由全局模型参数和集群偏差组成。
    """

    def __init__(self, client_id, data, base_model, device, lr=1e-4, weight_decay=1e-5):
        """
        Args:
            client_id (...): 客户端ID。
            data (...): 客户端本地数据 (包含节点分类 masks)。
            base_model (nn.Module): 基础模型架构 (GNN + Classifier)。
            device (torch.device): 设备。
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # 客户端的本地模型 (theta_i)
        self.local_model = base_model.to(device)

        # 用于存储上一轮接收到的全局模型参数 (theta_Global)，用于计算偏差
        self.global_state_dict = None

        # 用于存储客户端的集群ID (在 Server 聚类后确定)
        self.current_cluster_id = -1

        self.lr = lr
        self.weight_decay = weight_decay

    def get_optimizer(self):
        """为本地模型创建优化器。"""
        return optim.Adam(self.local_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    # --------------------------------------------------------------------------
    # 【核心 1】设置参数：初始化客户端模型为 (theta_Global + Delta_theta_Cluster)
    # --------------------------------------------------------------------------
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        接收服务器分发的参数，包括全局模型 theta_Global 和本客户端所属集群的偏差 Delta_theta_Cluster。

        Args:
            parameters (Dict): 包含 "global_model" 和 "cluster_bias" (集群偏差) 的参数。
        """

        # 1. 存储全局模型参数，用于计算后续的 Delta_theta_i
        # 使用深拷贝确保在本地训练时不改变原始全局参数
        self.global_state_dict = copy.deepcopy(parameters["global_model"])

        # 2. 计算本地模型的初始参数 theta_i = theta_Global + Delta_theta_Cluster, k

        # 加载全局模型到本地模型
        self.local_model.load_state_dict(self.global_state_dict)

        # 应用集群偏差 Delta_theta_Cluster, k
        cluster_bias = parameters["cluster_bias"]

        # 遍历模型参数，执行加法
        new_state_dict = self.local_model.state_dict()
        for key in new_state_dict.keys():
            if key in cluster_bias:
                # theta_i = theta_Global + Delta_theta_Cluster
                # 注意：这里直接将偏差加到加载了 theta_Global 的本地模型上
                new_state_dict[key] += cluster_bias[key].to(self.device)

        self.local_model.load_state_dict(new_state_dict)

        # 3. 记录当前的集群 ID (由 Server 决定并下发)
        self.current_cluster_id = parameters["cluster_id"]

    # --------------------------------------------------------------------------
    # 【核心 2】本地训练
    # --------------------------------------------------------------------------
    def local_train(self, epochs):

        optimizer = self.get_optimizer()
        self.local_model.train()

        train_mask = self.data.train_mask
        valid_labels_mask = (self.data.y >= 0)
        final_train_mask = train_mask & valid_labels_mask

        if final_train_mask.sum().item() == 0:
            print(f"Client {self.client_id} has no training nodes.")
            return 0.0

        for _ in range(epochs):
            optimizer.zero_grad()

            # 前向传播 (GNN + Classifier)
            logits = self.local_model(self.data.x, self.data.edge_index)

            train_logits = logits[final_train_mask]
            train_labels = self.data.y[final_train_mask]

            loss = self.criterion(train_logits, train_labels)

            loss.backward()
            optimizer.step()

        return loss.item()

    # --------------------------------------------------------------------------
    # 【核心 3】上传：返回个性化偏差 Delta_theta_i
    # --------------------------------------------------------------------------
    def get_trained_update(self):
        """
        返回训练后的个性化偏差 Delta_theta_i = theta_i - theta_Global。
        """
        if self.global_state_dict is None:
            raise ValueError("Global model parameters must be set before calculating update bias.")

        # 训练后的本地模型参数
        local_state = self.local_model.state_dict()

        # 初始化偏差字典
        personalized_bias = {}

        # 计算偏差 Delta_theta_i
        for key in local_state.keys():
            global_param = self.global_state_dict[key].to(self.device)
            local_param = local_state[key]

            # Delta_theta_i = theta_i - theta_Global
            bias = local_param - global_param

            # 使用深拷贝确保 Server 聚合时不会意外修改本地参数
            personalized_bias[key] = copy.deepcopy(bias)

        return {
            "personalized_bias": personalized_bias,  # Delta_theta_i
            "local_model": copy.deepcopy(local_state),  # theta_i (用于全局 FedAvg)
            "num_nodes": self.data.train_mask.sum().item()  # 用于加权聚合
        }

    # --------------------------------------------------------------------------
    # 评估：使用本地个性化模型 (theta_i) 评估
    # --------------------------------------------------------------------------
    def evaluate(self, use_test=False):
        """
        评估当前客户端的本地模型 (theta_i)。
        """
        self.local_model.eval()

        with torch.no_grad():
            logits = self.local_model(self.data.x, self.data.edge_index)

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
