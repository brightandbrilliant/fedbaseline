import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np


# 注意：sklearn.metrics 导入的位置通常在 evaluate 内部或文件顶部，这里放在文件顶部
# 确保您的环境中安装了 scikit-learn

class Client:
    def __init__(self, client_id, data: Data, encoder: nn.Module, classifier: nn.Module, device='cpu', lr=0.005,
                 weight_decay=1e-4):
        """
        FedAvg 客户端初始化，用于节点分类任务。

        Args:
            client_id (int): 客户端标识符。
            data (torch_geometric.data.Data): 客户端本地子图数据。
            encoder (nn.Module): GNN 编码器 (如 GCN/GraphSage)。
            classifier (nn.Module): ResMLP 分类头。
            device (str): 运行设备。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
        """
        self.client_id = client_id
        # 将数据移到指定设备
        self.data = data.to(device)
        self.device = device
        # 模型组件
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)

        # 优化器：优化编码器和分类头的所有参数
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # 关键：更换为节点分类的损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

        # 基础 Mask 检查
        if not hasattr(self.data, 'train_mask') or self.data.train_mask is None:
            raise AttributeError("客户端数据必须包含 'train_mask' 属性用于节点分类。")

    def train(self):
        """执行客户端本地训练的一步迭代，并跳过无效标签节点。"""
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad()

        # 1. 编码器生成节点嵌入
        z = self.encoder(self.data.x, self.data.edge_index)

        # 2. 分类器生成 Logits
        logits = self.classifier(z)

        # 3. 构造最终训练 Mask（核心：过滤无效标签）
        train_mask = self.data.train_mask

        # 过滤掉所有 y < 0 的无效标签（解决了 CUDA 错误）
        valid_labels_mask = (self.data.y >= 0)

        # 最终训练集 Mask：必须是训练节点 且 标签有效
        final_train_mask = train_mask & valid_labels_mask

        # 4. 提取 Logits 和 Labels
        train_logits = logits[final_train_mask]
        train_labels = self.data.y[final_train_mask]

        # 5. 健壮性检查：如果有效训练节点数为零，则跳过
        if train_labels.numel() == 0:
            # print(f"⚠️ Client {self.client_id}: 有效训练节点为零，跳过本轮训练。")
            return 0.0

        # 6. 计算损失
        loss = self.criterion(train_logits, train_labels)

        # 7. 反向传播与优化
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, use_test=False):
        """在本地验证集或测试集上评估模型性能，并跳过无效标签节点。"""
        self.encoder.eval()
        self.classifier.eval()

        # 1. 根据模式选择评估 mask
        if use_test:
            initial_mask = self.data.test_mask
        else:
            initial_mask = self.data.val_mask

        # 2. 构造最终评估 Mask（核心：过滤无效标签）
        valid_labels_mask = (self.data.y >= 0)

        # 最终评估 Mask：必须是评估节点 且 标签有效
        final_eval_mask = initial_mask & valid_labels_mask

        if final_eval_mask.sum().item() == 0:
            # print(f"⚠️ 客户端 {self.client_id}: 有效评估节点为零，跳过评估。")
            return 0.0, 0.0, 0.0, 0.0  # acc, recall, precision, f1

        with torch.no_grad():
            # 3. 编码器和分类器前向传播
            z = self.encoder(self.data.x, self.data.edge_index)
            logits = self.classifier(z)

            # 4. 提取评估集节点及其标签
            eval_logits = logits[final_eval_mask]
            eval_labels = self.data.y[final_eval_mask]

            # 5. 获取预测类别 (argmax)
            pred_labels = eval_logits.argmax(dim=1)

            # 转换为 numpy 数组进行 Scikit-learn 指标计算
            true_labels_np = eval_labels.cpu().numpy()
            pred_labels_np = pred_labels.cpu().numpy()

            # 6. 计算指标 (加权平均是多分类的常用标准)
            acc = accuracy_score(true_labels_np, pred_labels_np)
            recall = recall_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
            precision = precision_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
            f1 = f1_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)

        return acc, recall, precision, f1

    # --- 参数管理逻辑 (用于 FedAvg 聚合) ---

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_classifier_state(self):
        return self.classifier.state_dict()

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_classifier_state(self, state_dict):
        self.classifier.load_state_dict(state_dict)
