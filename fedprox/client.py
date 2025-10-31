import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import copy  # 需要用于深拷贝全局模型状态


class Client:
    def __init__(self, client_id, data: Data, encoder: nn.Module, classifier: nn.Module, device='cpu', lr=0.005,
                 weight_decay=1e-4, mu=0.0):  # <-- 关键修改 1: 添加 mu 参数
        """
        FedProx 客户端初始化，用于节点分类任务。

        Args:
            client_id (int): 客户端标识符。
            data (torch_geometric.data.Data): 客户端本地子图数据。
            encoder (nn.Module): GNN 编码器 (如 GCN/GraphSage)。
            classifier (nn.Module): ResMLP 分类头。
            device (str): 运行设备。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            mu (float): FedProx 的近端正则化参数，mu=0.0 时退化为 FedAvg。
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device

        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)

        # FedProx 参数
        self.mu = mu

        # 关键修改 2: 存储全局模型（用于近端项计算）
        # 我们需要两个独立的副本，分别用于编码器和分类器
        self.global_encoder_state = copy.deepcopy(encoder.state_dict())
        self.global_classifier_state = copy.deepcopy(classifier.state_dict())

        # 优化器：优化编码器和分类头的所有参数
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        if not hasattr(self.data, 'train_mask') or self.data.train_mask is None:
            raise AttributeError("客户端数据必须包含 'train_mask' 属性用于节点分类。")

    def calculate_proximal_term(self):
        """
        计算 FedProx 的近端项：
        (mu / 2) * || w_local - w_global ||^2
        """
        prox_loss = 0.0

        # 1. 编码器 (Encoder) 的近端项
        local_encoder_params = self.encoder.state_dict()
        for name in local_encoder_params:
            # 确保只对可学习参数（即有梯度）计算近端项，通常跳过 num_batches_tracked 等非参数
            if name in self.global_encoder_state:
                local_param = local_encoder_params[name]
                global_param = self.global_encoder_state[name].to(self.device)

                # 计算 L2 范数平方
                prox_loss += torch.pow((local_param - global_param), 2).sum()

        # 2. 分类器 (Classifier) 的近端项
        local_classifier_params = self.classifier.state_dict()
        for name in local_classifier_params:
            if name in self.global_classifier_state:
                local_param = local_classifier_params[name]
                global_param = self.global_classifier_state[name].to(self.device)

                prox_loss += torch.pow((local_param - global_param), 2).sum()

        return (self.mu / 2.0) * prox_loss

    def train(self):
        """执行客户端本地训练的一步迭代，加入近端损失项。"""
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad()

        # 1. 编码器生成节点嵌入
        z = self.encoder(self.data.x, self.data.edge_index)

        # 2. 分类器生成 Logits
        logits = self.classifier(z)

        # 3. 构造最终训练 Mask（过滤无效标签，与 FedAvg 相同）
        train_mask = self.data.train_mask
        valid_labels_mask = (self.data.y >= 0)
        final_train_mask = train_mask & valid_labels_mask

        train_logits = logits[final_train_mask]
        train_labels = self.data.y[final_train_mask]

        if train_labels.numel() == 0:
            return 0.0

        # 4. 计算本地损失 (节点分类 CrossEntropyLoss)
        local_loss = self.criterion(train_logits, train_labels)

        # 5. 计算近端项 (FedProx 核心)
        prox_term = self.calculate_proximal_term()

        # 6. 最终损失 = 本地损失 + 近端项
        loss = local_loss + prox_term

        # 7. 反向传播与优化
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ----------------------------------------------------
    # 评估逻辑 (Evaluate) 保持不变，因为它只需要计算性能指标，不涉及训练损失
    # ----------------------------------------------------
    def evaluate(self, use_test=False):
        """在本地验证集或测试集上评估模型性能，并跳过无效标签节点。"""
        self.encoder.eval()
        self.classifier.eval()

        if use_test:
            initial_mask = self.data.test_mask
        else:
            initial_mask = self.data.val_mask

        valid_labels_mask = (self.data.y >= 0)
        final_eval_mask = initial_mask & valid_labels_mask

        if final_eval_mask.sum().item() == 0:
            return 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)
            logits = self.classifier(z)

            eval_logits = logits[final_eval_mask]
            eval_labels = self.data.y[final_eval_mask]

            pred_labels = eval_logits.argmax(dim=1)

            true_labels_np = eval_labels.cpu().numpy()
            pred_labels_np = pred_labels.cpu().numpy()

            from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
            acc = accuracy_score(true_labels_np, pred_labels_np)
            recall = recall_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
            precision = precision_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
            f1 = f1_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)

        return acc, recall, precision, f1

    # ----------------------------------------------------
    # 参数管理逻辑 (FedProx 需要更新全局状态副本)
    # ----------------------------------------------------

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_classifier_state(self):
        return self.classifier.state_dict()

    def set_encoder_state(self, state_dict):
        # 更新本地模型参数
        self.encoder.load_state_dict(state_dict)
        # 关键修改 3: 更新全局模型参数的副本
        self.global_encoder_state = copy.deepcopy(state_dict)

    def set_classifier_state(self, state_dict):
        # 更新本地模型参数
        self.classifier.load_state_dict(state_dict)
        # 关键修改 3: 更新全局模型参数的副本
        self.global_classifier_state = copy.deepcopy(state_dict)
