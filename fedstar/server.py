import copy
import torch


class Server:
    def __init__(self, feature_encoder, structure_encoders, decoder, device):
        """
        初始化服务器端。
        - feature_encoder: 特征通道的全局编码器（GNN）
        - structure_encoders: 多个结构通道的全局编码器列表
        - decoder: 节点分类头（全局共享）
        - device: 计算设备
        """
        # 注意：这里我们将 decoder 视为节点分类头 (Classifier)
        self.feature_encoder = feature_encoder.to(device)
        self.structure_encoders = [enc.to(device) for enc in structure_encoders]
        self.decoder = decoder.to(device)
        self.device = device

    def get_global_parameters(self):
        """
        获取当前全局模型的参数。
        """
        return {
            "feature_encoder": copy.deepcopy(self.feature_encoder.state_dict()),
            "structure_encoders": [copy.deepcopy(enc.state_dict()) for enc in self.structure_encoders],
            "decoder": copy.deepcopy(self.decoder.state_dict())
        }

    def set_global_parameters(self, parameters):
        """
        设置全局模型的参数（通常在初始化或外部加载时使用）。
        """
        self.feature_encoder.load_state_dict(parameters["feature_encoder"])
        for enc, enc_params in zip(self.structure_encoders, parameters["structure_encoders"]):
            enc.load_state_dict(enc_params)
        self.decoder.load_state_dict(parameters["decoder"])

    def aggregate_feature_weights(self, selected_clients):
        """
        聚合特征通道编码器的参数（FedAvg）。
        """
        total_samples = sum(client.data.num_nodes for client in selected_clients)
        new_state = copy.deepcopy(self.feature_encoder.state_dict())

        for key in new_state.keys():
            new_state[key] = sum(
                (client.feature_encoder.state_dict()[key] * (client.data.num_nodes / total_samples))
                for client in selected_clients
            )

        self.feature_encoder.load_state_dict(new_state)

    def aggregate_structure_weights(self, selected_clients):
        """
        聚合多个结构通道编码器的参数（FedAvg）。
        """
        for idx in range(len(self.structure_encoders)):
            total_samples = sum(client.data.num_nodes for client in selected_clients)
            new_state = copy.deepcopy(self.structure_encoders[idx].state_dict())

            for key in new_state.keys():
                new_state[key] = sum(
                    (client.structure_encoders[idx].state_dict()[key] * (client.data.num_nodes / total_samples))
                    for client in selected_clients
                )

            self.structure_encoders[idx].load_state_dict(new_state)

    def aggregate_decoder_weights(self, selected_clients):
        """
        聚合分类头（原解码器）的参数（FedAvg）。
        """
        total_samples = sum(client.data.num_nodes for client in selected_clients)
        new_state = copy.deepcopy(self.decoder.state_dict())

        for key in new_state.keys():
            new_state[key] = sum(
                (client.decoder.state_dict()[key] * (client.data.num_nodes / total_samples))
                for client in selected_clients
            )

        self.decoder.load_state_dict(new_state)

    def aggregate_all_weights(self, selected_clients):
        """
        同时聚合特征通道、结构通道和分类头。
        """
        self.aggregate_feature_weights(selected_clients)
        self.aggregate_structure_weights(selected_clients)
        self.aggregate_decoder_weights(selected_clients)

    def distribute_parameters(self, clients):
        """
        将全局参数下发给所有客户端。
        """
        params = self.get_global_parameters()
        for client in clients:
            client.set_parameters(params)
            