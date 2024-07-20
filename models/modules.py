import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Multi-Layer Perceptron Classifier.
        :param input_dim: int, dimension of input
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        multi-layer perceptron classifier forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        # Tensor, shape (*, 80)
        x = self.dropout(self.act(self.fc1(x)))
        # Tensor, shape (*, 10)
        x = self.dropout(self.act(self.fc2(x)))
        # Tensor, shape (*, 1)
        return self.fc3(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor,
                neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor,
                neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features],
                                dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class MultiHeadAttention_Pool(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention_Pool, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor,
                neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor,
                neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features],
                                dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TransformerEncoder_Original(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder_Original, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0,
                                                                                                         1), inputs_key.transpose(
            0, 1), inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[
            0].transpose(0, 1)

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs


class FeedForwardNet_Pool(nn.Module):

    def __init__(self, kernel_size: int, num_channels, dropout: float):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet_Pool, self).__init__()
        self.kernel_size = kernel_size
        # self.kernel_size = kernel_size
        # # print(kernel_size, kernel_size // 2)
        # self.pool = nn.Sequential(
        # nn.AvgPool1d(1, stride=1, padding=1 // 2, count_include_pad=False),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2),
        #     nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,),
        #     nn.Conv1d(in_channels=1, out_channels=8, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,),
        #     nn.Dropout(dropout)
        # )

        self.kernel_total = nn.Parameter(torch.rand(1, 1, kernel_size*2-1), requires_grad=True)
        # self.kernel_previous = nn.Parameter(torch.rand(1, 1, kernel_size), requires_grad=True)
        # self.kernel_behind = nn.Parameter(torch.rand(1, 1, kernel_size), requires_grad=True)
        # nn.GELU(),
        # nn.Dropout(dropout),
        # nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False),
        # nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, delta_time: torch.Tensor=None):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        matrix_previous = []
        matrix_behind = []
        for i in range(self.kernel_size):
            rolled_tensor = torch.roll(x, shifts=i, dims=2)
            rolled_beh = torch.roll(x, shifts=-i, dims=2)
            rolled_tensor[:, :, :i] = 0
            rolled_beh[:, :, -i:] = 0
            matrix_previous.append(rolled_tensor)
            matrix_behind.append(rolled_beh)
        # delta_times = [delta_time]
        # for i in range(self.kernel_size):
        #     delt_rolled = torch.roll(delta_time, shifts=i, dims=1)
        #     rolled_tensor = torch.roll(x, shifts=i, dims=2)
        #     rolled_tensor[:, :, :i] = 0
        #     delt_rolled[:, :i] = 1e20
        #     matrix_total.append(rolled_tensor)
        #     delta_times.append(delta_times[-1]-delt_rolled)
        # matrix_total.append(x)
        # rolled_tensor = torch.roll(x, shifts=1, dims=2)
        # rolled_tensor[:, :, :1] = 0
        # matrix_total.append(rolled_tensor)
        # rolled_tensor = torch.roll(x, shifts=-1, dims=2)
        # rolled_tensor[:, :, -1:] = 0
        # matrix_total.append(rolled_tensor)
        matrix_total = torch.stack(matrix_previous, dim=-1).to(x.device)
        matrix_behind = torch.stack(matrix_behind[1:], dim=-1).to(x.device)
        # delta_times = torch.stack(delta_times[1:], dim=-1).to(x.device)
        # delta_times = torch.softmax(delta_times, dim=-1).unsqueeze(dim=1)
        ## 改为时间编码
        # average = (matrix_total * delta_times).sum(dim=-1)
        matrix_total = torch.cat([matrix_total, matrix_behind], dim=-1)
        # average_previous = (matrix_total * self.kernel_previous).sum(dim=-1)
        average = (matrix_total * self.kernel_total).sum(dim=-1)
        # average_behind = (matrix_behind * self.kernel_behind).sum(dim=-1)
        return average
        # return average_previous + average_behind

class FeedForwardNet_Time(nn.Module):

    def __init__(self, kernel_size: int, num_channels, dropout: float):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet_Time, self).__init__()
        self.kernel_size = kernel_size

        self.kernel_total = nn.Parameter(torch.rand(1, 1, kernel_size), requires_grad=True)
        # self.kernel_previous = nn.Parameter(torch.rand(1, 1, kernel_size), requires_grad=True)
        # self.kernel_behind = nn.Parameter(torch.rand(1, 1, kernel_size), requires_grad=True)
        # nn.GELU(),
        # nn.Dropout(dropout),
        # nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False),
        # nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, delta_time: torch.Tensor=None):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        matrix_previous = []
        delta_times = [delta_time]
        for i in range(self.kernel_size):
            rolled_tensor = torch.roll(x, shifts=i, dims=2)
            rolled_tensor[:, :, :i] = 0
            matrix_previous.append(rolled_tensor)
        # delta_times = [delta_time]
        for i in range(self.kernel_size):
            delt_rolled = torch.roll(delta_time, shifts=i, dims=1)
            delt_rolled[:, :i] = 1e20
            delta_times.append(delt_rolled - delta_times[0])  # 基准均是当前时刻
        # matrix_total.append(x)
        # rolled_tensor = torch.roll(x, shifts=1, dims=2)
        # rolled_tensor[:, :, :1] = 0
        # matrix_total.append(rolled_tensor)
        # rolled_tensor = torch.roll(x, shifts=-1, dims=2)
        # rolled_tensor[:, :, -1:] = 0
        # matrix_total.append(rolled_tensor)
        matrix_total = torch.stack(matrix_previous, dim=-1).to(x.device)
        delta_times = torch.stack(delta_times[1:], dim=-1).to(x.device)
        delta_times = torch.softmax(delta_times, dim=-1).unsqueeze(dim=1)
        ## 改为时间编码
        average_time = (matrix_total * delta_times).sum(dim=-1) # B * F * N
        average_previous = (matrix_total * self.kernel_total).sum(dim=-1)
        # average = (matrix_total * self.kernel_total).sum(dim=-1)
        # average_behind = (matrix_behind * self.kernel_behind).sum(dim=-1)
        average = torch.stack([average_time, average_previous], dim=-2) # B * F * 2 * N
        soft_average = torch.softmax(torch.matmul(average, average.transpose(2,3)), dim=-1)
        average = torch.matmul(soft_average, average).mean(dim=-2)
        # return average_previous + average_behind
        return average

class FeedForwardNet_Pre(nn.Module):

    def __init__(self, kernel_size: int, pre_kernel_size: int, num_channels, dropout: float):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet_Pre, self).__init__()
        self.kernel_size = kernel_size
        self.pre_kernel_size = pre_kernel_size
        self.random_R = nn.Parameter(torch.rand(num_channels, 10//2), requires_grad=False)
        self.kernel_total = nn.Parameter(torch.rand(1, 1, kernel_size-pre_kernel_size+1), requires_grad=True)
        # self.kernel_previous = nn.Parameter(torch.rand(1, 1, kernel_size), requires_grad=True)
        # self.kernel_behind = nn.Parameter(torch.rand(1, 1, kernel_size), requires_grad=True)
        # nn.GELU(),
        # nn.Dropout(dropout),
        # nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False),
        # nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        seq_rep = x.transpose(1,2)
        hash_rep = torch.matmul(seq_rep, self.random_R)
        total_hash = torch.cat([hash_rep, -hash_rep], dim=-1)
        hash_index = torch.argmax(total_hash, dim=-1)
        _, indices = torch.sort(hash_index, dim=-1)
        indices = indices.unsqueeze(dim=-1)
        new_x = torch.gather(seq_rep, 1, indices).transpose(1,2)

        matrix_previous = []
        for i in np.arange(self.pre_kernel_size, self.kernel_size):
            rolled_tensor = torch.roll(new_x, shifts=i, dims=2)
            rolled_tensor[:, :, :i] = 0
            matrix_previous.append(rolled_tensor)
        matrix_previous.append(new_x)
        matrix_total = torch.stack(matrix_previous, dim=-1).to(x.device)
        average_previous = (matrix_total * self.kernel_total).sum(dim=-1)
        return average_previous


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, output_dim: int, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout
        self.output_dim = output_dim

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=output_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)


class TransformerEncoderBlock(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_kernel_size: int,
                 dropout: float, num_heads: int = 1, channel_kernel_size: int = 1,
                 token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(TransformerEncoderBlock, self).__init__()
        self.num_tokens = num_tokens
        self.num_channel = num_channels
        self.dropout = dropout

        if isinstance(token_kernel_size, list):
            self.transformer_layers = nn.ModuleList()
            for i, kernel_size in enumerate(token_kernel_size):
                self.transformer_layers.append(TransformerEncoder(num_tokens=num_tokens,
                    token_kernel_size=kernel_size, num_channels=num_channels, pre_kernel_size=0 if i==0 else token_kernel_size[i-1],
                                                     dropout=dropout))
        elif isinstance(token_kernel_size, int):
            self.transformer_layers=nn.ModuleList([TransformerEncoder(num_tokens=num_tokens,
                    token_kernel_size=token_kernel_size, num_channels=num_channels,
                                                     dropout=dropout)])
        self.drop = nn.Dropout(self.dropout)

    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        for encoder in self.transformer_layers:
            input_tensor = self.drop(encoder(input_tensor))
            # input_tensor = encoder(input_tensor)

        return input_tensor



class TransformerEncoder(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_kernel_size: int,
                 dropout: float, pre_kernel_size: int = 0, num_heads: int = 1, channel_kernel_size: int = 1,
                 token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.num_tokens = num_tokens
        self.num_channel = num_channels
        self.token_norm = nn.LayerNorm(num_tokens)

        self.token_feedforward = FeedForwardNet_Pre(kernel_size=token_kernel_size, num_channels=num_channels,
                                                    pre_kernel_size=pre_kernel_size,
                                                              dropout=dropout)

        # self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
        #                                         dropout=dropout, output_dim=num_tokens)
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels,
                                                  dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout, output_dim=num_channels)

    # def forward(self, input_tensor: torch.Tensor, delta_times: torch.Tensor=None):
    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # print(input_tensor.shape)
        # Tensor, shape (batch_size, num_channels, num_tokens)
        batch_size, num_tokens, num_channels = input_tensor.shape
        # hidden_tensor = input_tensor.permute(0, 2, 1).reshape(batch_size * num_channels, 1, num_tokens) # todo: 不加norm效果更好一些
        hidden_tensor = input_tensor.permute(0, 2, 1)

        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)

        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor


# # pool
# class TransformerEncoder(nn.Module):
#
#     def __init__(self, attention_dim: int, num_heads: int, pool_kernel_size: 'int|list', dropout: float = 0.1):
#         """
#         Transformer encoder.
#         :param attention_dim: int, dimension of the attention vector
#         :param num_heads: int, number of attention heads
#         :param dropout: float, dropout rate
#         """
#         super(TransformerEncoder, self).__init__()
#         # use the MultiheadAttention implemented by PyTorch
#         # self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)
#
#         # replace mha with avarage pooling
#         if type(pool_kernel_size) is int:
#             self.token_mixer = Pooling(kernel_size=pool_kernel_size)
#         else:
#             self.token_mixer = nn.ModuleList([
#                 Pooling(kernel_size=i)
#                 for i in pool_kernel_size])
#             self.dense = nn.Linear(in_features=len(pool_kernel_size) * attention_dim,
#                                    out_features=attention_dim)
#
#         self.dropout = nn.Dropout(dropout)
#
#         # ǰ
#         # self.linear_layers = nn.ModuleList([
#         #     nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
#         #     nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
#         # ])
#         self.norm_layers = nn.ModuleList([
#             nn.LayerNorm(attention_dim),
#             nn.LayerNorm(attention_dim)
#         ])
#
#     def forward(self, inputs: torch.Tensor):
#         """
#         encode the inputs by Transformer encoder
#         :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
#         :return:
#         """
#         ln_inputs = self.norm_layers[0](inputs)
#
#         if type(self.token_mixer) is nn.ModuleList:
#             # hidden_states = self.token_mixer[0](ln_inputs)
#             # for i in range(1,len(self.token_mixer)):
#             #     hidden_states += self.token_mixer[i](ln_inputs)  # hiden = hiden + x
#             hidden_states = []
#             for i in range(0, len(self.token_mixer)):
#                 hidden_states.append(self.token_mixer[i](ln_inputs))
#             hidden_states = self.dense(torch.cat((hidden_states), dim=2))
#             # print(torch.stack(hidden_states,dim=3).shape, self.dense(torch.stack(hidden_states,dim=3)).shape, torch.squeeze(self.dense(torch.stack(hidden_states,dim=3))).shape)
#             # hidden_states = torch.squeeze(self.dense(torch.stack(hidden_states,dim=3)))
#         else:
#             hidden_states = self.token_mixer(ln_inputs)
#
#         # # Tensor, shape (batch_size, num_patches, self.attention_dim)
#         # outputs = inputs + self.dropout(hidden_states)
#         # # Tensor, shape (batch_size, num_patches, self.attention_dim)
#         # hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
#         # # Tensor, shape (batch_size, num_patches, self.attention_dim)
#         # outputs = outputs + self.dropout(hidden_states)
#
#         outputs = inputs + self.dropout(hidden_states)
#         # outputs = self.norm_layers[1](outputs) + self.dropout(hidden_states)
#         outputs = self.norm_layers[1](outputs)
#
#         return outputs


class TransformerEncoder_1_pool(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, pool_kernel_size: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder_1_pool, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        # self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        # replace mha with avarage pooling
        self.token_mixer = Pooling(kernel_size=pool_kernel_size)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        ln_inputs = self.norm_layers[0](inputs)
        hidden_states = self.token_mixer(ln_inputs)

        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)

        return outputs


# Pool_0
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
        pool_size == 2i+1, padd == i;
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.pool = nn.AvgPool1d(
            kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.pool(x).transpose(2, 1)


# MultiPooling_1
class MultiPooling_1(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, pool_kernel_size: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiPooling_1, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        # self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        # self.head_dim = self.query_dim // num_heads

        self.pool = nn.AvgPool1d(kernel_size=self.edge_feat_dim + 1,
                                 stride=1, padding=0, count_include_pad=False)
        #
        # self.query_projection = nn.Linear(self.query_dim, self.node_feat_dim + self.time_feat_dim, bias=False)
        # self.key_projection = nn.Linear(self.key_dim, self.node_feat_dim + self.time_feat_dim, bias=False)
        # self.value_projection = nn.Linear(self.key_dim, self.node_feat_dim + self.time_feat_dim, bias=False)

        # self.scaling_factor = self.head_dim ** -0.5
        self.layer_norm_0 = nn.LayerNorm(self.query_dim)
        self.layer_norm_1 = nn.LayerNorm(self.query_dim)

        # self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor,
                neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor,
                neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)
        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features],
                                dim=2)

        key = torch.unsqueeze(torch.mean(self.pool(key), dim=1), dim=1)
        value = torch.unsqueeze(torch.mean(self.pool(value), dim=1), dim=1)
        # value = torch.mean(self.pool(value),dim=1)

        # # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim)
        # query = self.query_projection(query)
        # # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        # key = self.key_projection(key)
        # # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        # value = self.value_projection(value)
        # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.layer_norm_0(query + key + value))
        # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim)
        output = torch.unsqueeze(torch.mean(output, dim=1), dim=1)
        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm_1(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        return output, 0


# MultiPooling_0
class MultiPooling_0(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, pool_kernel_size: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiPooling_0, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        # self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        # self.head_dim = self.query_dim // num_heads

        self.pool = nn.AvgPool1d(
            pool_kernel_size, stride=1, padding=pool_kernel_size // 2, count_include_pad=False)

        self.query_projection = nn.Linear(self.query_dim, self.node_feat_dim + self.time_feat_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, self.node_feat_dim + self.time_feat_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, self.node_feat_dim + self.time_feat_dim, bias=False)

        # self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)

        # self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor,
                neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor,
                neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)
        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features],
                                dim=2)

        # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim)
        query = self.query_projection(query)
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        key = self.key_projection(key)
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        value = self.value_projection(value)
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        output = self.dropout(query + key + value)
        # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim)
        output = torch.unsqueeze(torch.mean(output, dim=1), dim=1)
        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        return output, 0


# MultiPooling_2
class MultiPooling_2(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, pool_kernel_size: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiPooling_2, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        # self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        # self.head_dim = self.query_dim // num_heads

        self.pool = nn.AvgPool1d(kernel_size=self.edge_feat_dim + 1,
                                 stride=1, padding=0, count_include_pad=False)
        #
        # self.query_projection = nn.Linear(self.query_dim, self.node_feat_dim + self.time_feat_dim, bias=False)
        # self.key_projection = nn.Linear(self.key_dim, self.node_feat_dim + self.time_feat_dim, bias=False)
        # self.value_projection = nn.Linear(self.key_dim, self.node_feat_dim + self.time_feat_dim, bias=False)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=self.node_feat_dim + self.time_feat_dim,
                      out_features=4 * (self.node_feat_dim + self.time_feat_dim)),
            nn.Linear(in_features=4 * (self.node_feat_dim + self.time_feat_dim),
                      out_features=(self.node_feat_dim + self.time_feat_dim))
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim),
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim)
        ])

        # self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor,
                neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor,
                neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)
        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features],
                                dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = torch.unsqueeze(torch.mean(self.pool(key), dim=1), dim=1)
        value = torch.unsqueeze(torch.mean(self.pool(value), dim=1), dim=1)

        # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim)
        outputs = self.norm_layers[0](query + key + value + self.dropout(query + key + value))
        # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        outputs = outputs.squeeze(dim=1)
        return outputs, 0


# MultiPooling_3
class MultiPooling_3(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, pool_kernel_size: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiPooling_3, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        # self.head_dim = self.query_dim // num_heads

        self.pool = nn.AvgPool1d(
            pool_kernel_size, stride=1, padding=pool_kernel_size // 2, count_include_pad=False)

        self.query_projection = nn.Linear(self.query_dim, self.node_feat_dim + self.time_feat_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, self.node_feat_dim + self.time_feat_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, self.node_feat_dim + self.time_feat_dim, bias=False)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=self.node_feat_dim + self.time_feat_dim,
                      out_features=4 * (self.node_feat_dim + self.time_feat_dim)),
            nn.Linear(in_features=4 * (self.node_feat_dim + self.time_feat_dim),
                      out_features=(self.node_feat_dim + self.time_feat_dim))
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim),
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim)
        ])

        # self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor,
                neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor,
                neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)
        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features],
                                dim=2)

        # Tensor, shape (batch_size,1, node_feat_dim + time_feat_dim) + (batch_size, num_neighbors, edge_feat_dim)
        # [query, key] - �������е���Ϣ u:fn,fe,ft
        # b*1*(fn+ft), b*n_b*(fn+fe+ft)
        # b*n_b*(fn+fe+ft)
        query = self.query_projection(query)
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        key = self.key_projection(key)
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        value = self.value_projection(value)
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        outputs = self.norm_layers[0](query + key + value + self.dropout(query + key + value))
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        outputs = torch.mean(outputs, dim=1)

        return outputs, 0


# MultiPooling_4
class MultiPooling(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, pool_kernel_size: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiPooling, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.pool = nn.AvgPool1d(
            pool_kernel_size, stride=1, padding=pool_kernel_size // 2, count_include_pad=False)

        self.query_projection = nn.Linear(self.query_dim, self.num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, self.num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, self.num_heads * self.head_dim, bias=False)

        self.condense = nn.Linear(self.num_heads * self.head_dim, self.node_feat_dim + self.time_feat_dim, bias=False)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=self.node_feat_dim + self.time_feat_dim,
                      out_features=4 * (self.node_feat_dim + self.time_feat_dim)),
            nn.Linear(in_features=4 * (self.node_feat_dim + self.time_feat_dim),
                      out_features=(self.node_feat_dim + self.time_feat_dim))
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim),
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim)
        ])

        # self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor,
                neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor,
                neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = torch.cat([node_features, node_time_features], dim=2)

        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features],
                                dim=2)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        outputs = (query + key + value).permute(0, 2, 1, 3).flatten(start_dim=2)

        outputs = self.condense(outputs)

        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        outputs = self.norm_layers[0](outputs + self.dropout(outputs))
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size,num_neighbors, node_feat_dim + time_feat_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        outputs = torch.mean(outputs, dim=1)

        return outputs, 0


# MultiPooling_4_1
class MultiPooling_general_1(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, pool_kernel_size: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiPooling_general_1, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.pool = nn.AvgPool1d(
            pool_kernel_size, stride=1, padding=pool_kernel_size // 2, count_include_pad=False)

        self.query_projection = nn.Linear(self.query_dim, self.num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, self.num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, self.num_heads * self.head_dim, bias=False)

        self.condense = nn.Linear(self.num_heads * self.head_dim, self.node_feat_dim + self.time_feat_dim, bias=False)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=self.node_feat_dim + self.time_feat_dim,
                      out_features=4 * (self.node_feat_dim + self.time_feat_dim)),
            nn.Linear(in_features=4 * (self.node_feat_dim + self.time_feat_dim),
                      out_features=(self.node_feat_dim + self.time_feat_dim))
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim),
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim)
        ])

        # self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor = None,
                neighbor_node_features: torch.Tensor = None,
                neighbor_node_time_features: torch.Tensor = None, neighbor_node_edge_features: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if len(node_features.shape) == 2:
            # Tensor, shape (batch_size, 1, node_feat_dim)
            node_features = torch.unsqueeze(node_features, dim=1)

        if node_time_features != None:
            # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
            query = torch.cat([node_features, node_time_features], dim=2)
        else:
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            query = node_features

        # shape (batch_size, , num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        if neighbor_node_features != None:
            if neighbor_node_edge_features != None and neighbor_node_time_features != None:
                # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
                key = value = torch.cat(
                    [neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
            else:
                # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
                key = value = neighbor_node_features

            # Tensor, shape (batch_size, , num_heads, self.head_dim)
            key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
            # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
            value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

            # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
            query = query.permute(0, 2, 1, 3)
            # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
            key = key.permute(0, 2, 1, 3)
            # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
            value = value.permute(0, 2, 1, 3)

            # Tensor, shape (batch_size, num_heads, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
            outputs = (query + key + value).permute(0, 2, 1, 3).flatten(start_dim=2)
        else:
            outputs = self.pool(query.flatten(start_dim=2).transpose(1, 2)).transpose(1, 2)

        outputs = self.condense(outputs)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
        outputs = self.norm_layers[0](outputs + self.dropout(outputs))
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        outputs = torch.mean(outputs, dim=1)

        if len(node_features.shape) != 2:
            # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
            return torch.unsqueeze(outputs, dim=1)
        else:
            return outputs, 0


# MultiPooling_4_2
class MultiPooling_general_2(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, pool_kernel_size: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiPooling_general_2, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.pool = nn.AvgPool1d(
            pool_kernel_size, stride=1, padding=pool_kernel_size // 2, count_include_pad=False)

        self.query_projection = nn.Linear(self.query_dim, self.num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, self.num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, self.num_heads * self.head_dim, bias=False)

        self.condense = nn.Linear(self.num_heads * self.head_dim, self.node_feat_dim + self.time_feat_dim, bias=False)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=self.node_feat_dim + self.time_feat_dim,
                      out_features=4 * (self.node_feat_dim + self.time_feat_dim)),
            nn.Linear(in_features=4 * (self.node_feat_dim + self.time_feat_dim),
                      out_features=(self.node_feat_dim + self.time_feat_dim))
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim),
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim)
        ])

        # self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor = None,
                neighbor_node_features: torch.Tensor = None,
                neighbor_node_time_features: torch.Tensor = None, neighbor_node_edge_features: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if len(node_features.shape) == 2:
            # Tensor, shape (batch_size, 1, node_feat_dim)
            node_features = torch.unsqueeze(node_features, dim=1)

        if node_time_features != None:
            # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
            query = residual = torch.cat([node_features, node_time_features], dim=2)
        else:
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            query = residual = node_features

        # shape (batch_size, neighbor_nums, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        if neighbor_node_features != None:
            if neighbor_node_edge_features != None and neighbor_node_time_features != None:
                # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
                key = value = torch.cat(
                    [neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
            else:
                # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
                key = value = neighbor_node_features

            # Tensor, shape (batch_size, , num_heads, self.head_dim)
            key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
            # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
            value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

            # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
            query = query.permute(0, 2, 1, 3)
            # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
            key = key.permute(0, 2, 1, 3)
            # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
            value = value.permute(0, 2, 1, 3)

            # Tensor, shape (batch_size, num_heads, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
            outputs = (query + key + value).permute(0, 2, 1, 3).flatten(start_dim=2)

            outputs = self.condense(outputs)

            # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
            outputs = self.norm_layers[0](residual + self.dropout(outputs))
            # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
            hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
            # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
            outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

            # outputs = self.pool(outputs.transpose(1,2)).transpose(1,2)   Ч����
            # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
            outputs = torch.mean(outputs, dim=1)

            # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
            outputs = torch.unsqueeze(outputs, dim=1)
        else:
            # Tensor, shape (batch_size, neighbor_nums, node_feat_dim + time_feat_dim)
            outputs = self.pool(query.flatten(start_dim=2).transpose(1, 2)).transpose(1, 2)

            outputs = self.condense(outputs)

            # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
            outputs = self.norm_layers[0](residual + self.dropout(outputs))
            # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
            hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
            # Tensor, shape (batch_size, num_neighbors, node_feat_dim + time_feat_dim)
            outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs


# MultiPooling_4_3
class MultiPooling_general(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int, pool_kernel_size: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiPooling_general, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.pool = nn.AvgPool1d(
            pool_kernel_size, stride=1, padding=pool_kernel_size // 2, count_include_pad=False)

        # self.query_projection = nn.Linear(self.query_dim, self.num_heads * self.head_dim, bias=False)
        # self.key_projection = nn.Linear(self.key_dim, self.num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, self.num_heads * self.head_dim, bias=False)

        self.condense = nn.Linear(self.num_heads * self.head_dim, self.node_feat_dim + self.time_feat_dim, bias=False)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=self.node_feat_dim + self.time_feat_dim,
                      out_features=4 * (self.node_feat_dim + self.time_feat_dim)),
            nn.Linear(in_features=4 * (self.node_feat_dim + self.time_feat_dim),
                      out_features=(self.node_feat_dim + self.time_feat_dim))
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim),
            nn.LayerNorm(self.node_feat_dim + self.time_feat_dim)
        ])

        # self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor = None,
                neighbor_node_features: torch.Tensor = None,
                neighbor_node_time_features: torch.Tensor = None, neighbor_node_edge_features: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if neighbor_node_edge_features != None and neighbor_node_time_features != None:
            # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
            value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        else:
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            value = neighbor_node_features

        # Tensor, shape (batch_size, num_neighbors, num_heads * self.head_dim)
        value = self.value_projection(value)

        # Tensor, shape (batch_size, num_neighbors, num_heads * self.head_dim)
        value = self.pool(value.transpose(1, 2)).transpose(2, 1)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        outputs = torch.mean(value, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        outputs = torch.unsqueeze(outputs, dim=1)

        return outputs
