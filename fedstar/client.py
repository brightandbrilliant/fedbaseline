import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling  # <--- 尽管导入了，但在 train/eval 中不再使用
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class Client:
    def __init__(self, client_id, data, feature_encoder, structure_encoders, decoder, device, lr=1e-4,
                 weight_decay=1e-5):
        self.client_id = client_id
        self.data = data.to(device)
        self.feature_encoder = feature_encoder.to(device)
        self.structure_encoders = [enc.to(device) for enc in structure_encoders]
        self.decoder = decoder.to(device)  # 在节点分类中，decoder 现充当分类头 (Classifier)
        self.device = device

        # ----------------------------------------------------------------------
        # 【修改点 1】优化器：仅优化 feature_encoder (GNN) 和 decoder (分类头) 的参数。
        # 结构编码器（structure_encoders）的参数被视为固定的特征提取器，不参与训练。
        # ----------------------------------------------------------------------
        params = list(self.feature_encoder.parameters()) + list(self.decoder.parameters())

        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()  # <--- 新增交叉熵损失函数

        # 初始化结构特征缓存，并使用 no_grad 确保高效
        self.extract_structure_features()

    def get_parameters(self):
        return {
            "feature_encoder": self.feature_encoder.state_dict(),
            "structure_encoders": [enc.state_dict() for enc in self.structure_encoders],
            "decoder": self.decoder.state_dict()
        }

    def set_parameters(self, parameters):
        self.feature_encoder.load_state_dict(parameters["feature_encoder"])
        for enc, enc_params in zip(self.structure_encoders, parameters["structure_encoders"]):
            enc.load_state_dict(enc_params)
        self.decoder.load_state_dict(parameters["decoder"])

    def extract_structure_features(self):
        """
        提取结构特征，并缓存到 self.data.structure_x。
        【修改点 2】使用 torch.no_grad() 提高效率。
        """
        struct_features = []
        with torch.no_grad():
            for enc in self.structure_encoders:
                # 确保在提取特征时结构编码器处于评估模式（虽然我们不再训练它）
                enc.eval()
                struct_features.append(enc(self.data).to(self.device))

        self.data.structure_x = torch.cat(struct_features, dim=1)  # [N, sum(feature_dims)]

    def local_train(self, epochs):
        self.feature_encoder.train()
        # 结构编码器不再切换模式，因为它不参与优化 (已在 __init__ 中排除)
        self.decoder.train()

        # 缓存的结构特征 (固定)
        z_struct = self.data.structure_x

        # 【修改点 3】定义训练节点掩码
        train_mask = self.data.train_mask
        valid_labels_mask = (self.data.y >= 0)
        final_train_mask = train_mask & valid_labels_mask

        if final_train_mask.sum().item() == 0:
            return 0.0  # 无有效训练节点

        for _ in range(epochs):
            self.optimizer.zero_grad()

            # 1. 特征通道前向传播
            z_feat = self.feature_encoder(self.data.x, self.data.edge_index)  # [N, d_f]

            # 2. 融合特征和结构
            z = torch.cat([z_feat, z_struct], dim=1)  # [N, d_f + sum(d_i)]

            # 3. 【修改点 4】节点分类前向传播：
            # decoder (分类头) 接受融合特征 z，输出节点 logits [N, num_classes]
            logits = self.decoder(z)

            # 4. 【修改点 5】应用训练掩码和交叉熵损失
            train_logits = logits[final_train_mask]
            train_labels = self.data.y[final_train_mask]

            # 使用交叉熵损失
            loss = self.criterion(train_logits, train_labels)

            # 5. 反向传播
            loss.backward()
            self.optimizer.step()

        return loss.item()  # 返回最后一轮的损失

    def evaluate(self, use_test=False):
        """
        在验证集或测试集上评估当前客户端模型（节点分类版本）。
        """
        self.feature_encoder.eval()
        for enc in self.structure_encoders:
            enc.eval()
        self.decoder.eval()

        # 【修改点 6】评估时直接使用缓存的结构特征
        z_struct = self.data.structure_x

        with torch.no_grad():
            # 1. 特征通道
            z_feat = self.feature_encoder(self.data.x, self.data.edge_index)

            # 2. 融合
            z = torch.cat([z_feat, z_struct], dim=1)

            # 3. 节点分类前向传播
            logits = self.decoder(z)

            # 4. 【修改点 7】选择评估集掩码
            if use_test:
                initial_mask = self.data.test_mask
            else:
                initial_mask = self.data.val_mask

            # 确保只对有效标签的节点进行评估
            valid_labels_mask = (self.data.y >= 0)
            final_eval_mask = initial_mask & valid_labels_mask

            if final_eval_mask.sum().item() == 0:
                return 0.0, 0.0, 0.0, 0.0  # 无有效评估节点

            eval_logits = logits[final_eval_mask]
            y_true = self.data.y[final_eval_mask]

            # 5. 计算预测标签
            y_pred_labels = eval_logits.argmax(dim=1)

            # 6. 【修改点 8】计算多分类评估指标
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred_labels.cpu().numpy()

            acc = accuracy_score(y_true_np, y_pred_np)
            # 使用 'weighted' 平均来处理类别不平衡
            recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
            f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)

        return acc, recall, precision, f1
