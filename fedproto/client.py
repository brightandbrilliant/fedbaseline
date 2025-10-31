import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np


# 确保您的环境中安装了 scikit-learn

class Client:
    def __init__(self, client_id, data: Data, encoder: nn.Module, classifier: nn.Module,
                 device='cpu', lr=0.005, weight_decay=1e-4,
                 lambda_proto=1.0, num_classes=None):  # <-- 关键修改 1: 添加 FedProto 参数
        """
        FedProto 客户端初始化，用于节点分类任务。

        Args:
            client_id (int): 客户端标识符。
            data (...): 客户端本地子图数据。
            encoder (...): GNN 编码器。
            classifier (...): ResMLP 分类头。
            device (str): 运行设备。
            lambda_proto (float): FedProto 的原型正则化权重。
            num_classes (int): 数据集总类别数。
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device

        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)

        # FedProto 参数
        self.lambda_proto = lambda_proto
        self.num_classes = num_classes

        # 关键修改 2: 存储全局原型 (在 main.py 首次同步后才会有实际值)
        # 初始化为 None 或零向量，具体取决于 num_classes
        if num_classes is None:
            raise ValueError("num_classes 必须在 Client 初始化时提供给 FedProto。")

        # 假设原型维度等于编码器输出维度
        prototype_dim = self.encoder.output_dim
        self.global_prototypes = torch.zeros(num_classes, prototype_dim).to(device)

        # 优化器：优化编码器和分类头的所有参数
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        if not hasattr(self.data, 'train_mask') or self.data.train_mask is None:
            raise AttributeError("客户端数据必须包含 'train_mask' 属性用于节点分类。")

    def get_local_prototypes(self):
        """
        计算本地数据上每个类别的特征原型（类中心）。

        返回: dict, {class_id: prototype_tensor}
              list, [(class_id, count)] 用于聚合权重
        """
        self.encoder.eval()

        # 1. 使用编码器获取所有节点的特征嵌入
        with torch.no_grad():
            # z 已经是无梯度，不会影响后续训练
            z = self.encoder(self.data.x, self.data.edge_index)

            # 2. 构造本地训练集的有效掩码（与 train 保持一致）
        train_mask = self.data.train_mask
        valid_labels_mask = (self.data.y >= 0)
        final_train_mask = train_mask & valid_labels_mask

        # 提取有效的嵌入和标签
        train_z = z[final_train_mask]
        train_labels = self.data.y[final_train_mask]

        local_prototypes = {}
        prototype_counts = []

        # 3. 遍历所有类别，计算原型
        for class_id in range(self.num_classes):
            # 获取属于当前类别的节点嵌入
            class_mask = (train_labels == class_id)
            class_z = train_z[class_mask]

            num_samples = class_z.size(0)

            if num_samples > 0:
                # 原型 = 该类别所有嵌入的平均值
                prototype = class_z.mean(dim=0)
                local_prototypes[class_id] = prototype
                prototype_counts.append((class_id, num_samples))

        return local_prototypes, prototype_counts

    def calculate_proto_loss(self, local_prototypes):
        """
        计算 FedProto 的原型损失项：
        L_proto = sum_c || P_local^c - P_global^c ||^2
        """
        proto_loss = 0.0

        # 遍历本地计算出的所有原型
        for class_id, local_proto in local_prototypes.items():
            # 获取对应的全局原型
            global_proto = self.global_prototypes[class_id].to(self.device)

            # 计算 L2 范数平方
            # 使用 detach() 确保只有本地原型通过梯度更新
            proto_loss += torch.pow((local_proto - global_proto.detach()), 2).sum()

        return self.lambda_proto * proto_loss

    def train(self):
        """执行客户端本地训练的一步迭代，加入原型损失项。"""
        self.encoder.train()
        self.classifier.train()
        self.optimizer.zero_grad()

        # 1. 前向传播
        z = self.encoder(self.data.x, self.data.edge_index)
        logits = self.classifier(z)

        # 2. 构造最终训练 Mask（过滤无效标签）
        train_mask = self.data.train_mask
        valid_labels_mask = (self.data.y >= 0)
        final_train_mask = train_mask & valid_labels_mask

        train_logits = logits[final_train_mask]
        train_labels = self.data.y[final_train_mask]

        if train_labels.numel() == 0:
            return 0.0

        # 3. 计算本地分类损失 (L_CE)
        local_ce_loss = self.criterion(train_logits, train_labels)

        # 4. 计算本地原型 (需要在当前模型参数下计算)
        local_prototypes, _ = self.get_local_prototypes()

        # 5. 计算 FedProto 原型损失 (L_proto)
        # 注意：此处使用的 local_prototypes 依赖于当前的 encoder 参数
        proto_term = self.calculate_proto_loss(local_prototypes)

        # 6. 最终损失 = L_CE + L_proto
        loss = local_ce_loss + proto_term

        # 7. 反向传播与优化
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ----------------------------------------------------
    # 评估逻辑 (Evaluate) 保持不变
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
    # 参数管理逻辑 (新增原型同步方法)
    # ----------------------------------------------------

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_classifier_state(self):
        return self.classifier.state_dict()

    def get_global_prototypes(self):
        # 仅用于初始化，实际通信是 local prototypes
        return self.global_prototypes

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_classifier_state(self, state_dict):
        self.classifier.load_state_dict(state_dict)

    def set_global_prototypes(self, prototypes):
        """设置全局原型，用于本地正则化"""
        # 注意：这里接收的 prototypes 已经是聚合后的全局结果
        self.global_prototypes = prototypes.to(self.device)
        