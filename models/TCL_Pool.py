import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder, TransformerEncoder, TransformerEncoder_Original, TransformerEncoderBlock
from utils.utils import NeighborSampler


class TCL_Pool(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler, pool_kernel_size: int,
                num_neighbors: int,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, num_depths: int = 20, dropout: float = 0.1, device: str = 'cpu'):
        """
        TCL model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param num_depths: int, number of depths, identical to the number of sampled neighbors plus 1 (involving the target node)
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(TCL_Pool, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_depths = num_depths
        self.dropout = dropout
        self.device = device

        self.num_neighbors = num_neighbors

        self.pool_kernel_size = pool_kernel_size

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.depth_embedding = nn.Embedding(num_embeddings=num_depths, embedding_dim=self.node_feat_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.node_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.node_feat_dim, bias=True)
        })

        self.transformers = nn.ModuleList([
            # TransformerEncoder(attention_dim=self.node_feat_dim, dropout=self.dropout, num_heads=self.num_heads, pool_kernel_size=self.pool_kernel_size)
            TransformerEncoderBlock(num_tokens=self.num_neighbors+1,
                               num_channels=self.node_feat_dim,
                               token_kernel_size=pool_kernel_size,
                               dropout=dropout)
            for _ in range(self.num_layers)
        ])

        self.cross_transformers = nn.ModuleList([
            TransformerEncoder_Original(attention_dim=self.node_feat_dim, dropout=self.dropout, num_heads=self.num_heads)
            # TransformerEncoder(num_tokens=self.num_neighbors+1,
            #                    num_channels=self.node_feat_dim,
            #                    token_kernel_size=pool_kernel_size,
            #                    dropout=dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # get temporal neighbors of source nodes, including neighbor ids, edge ids and time information
        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors)
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # get temporal neighbors of destination nodes, including neighbor ids, edge ids and time information
        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors)
        dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate((np.zeros((len(src_node_ids), 1)).astype(np.longlong), src_neighbor_edge_ids), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)

        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_node_ids = np.concatenate((dst_node_ids[:, np.newaxis], dst_neighbor_node_ids), axis=1)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_edge_ids = np.concatenate((np.zeros((len(dst_node_ids), 1)).astype(np.longlong), dst_neighbor_edge_ids), axis=1)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], dst_neighbor_times), axis=1)

        # pad the features of the sequence of source and destination nodes
        # src_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # src_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # src_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # src_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_neighbor_depth_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_neighbor_node_ids,
                              nodes_edge_ids=src_neighbor_edge_ids, nodes_neighbor_times=src_neighbor_times, time_encoder=self.time_encoder)

        # dst_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # dst_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # dst_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # dst_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, dst_nodes_neighbor_depth_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=dst_neighbor_node_ids,
                              nodes_edge_ids=dst_neighbor_edge_ids, nodes_neighbor_times=dst_neighbor_times, time_encoder=self.time_encoder)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_nodes_neighbor_node_raw_features)
        dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        dst_nodes_neighbor_time_features = self.projection_layer['time'](dst_nodes_neighbor_time_features)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        src_node_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features + src_nodes_neighbor_depth_features
        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        dst_node_features = dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features + dst_nodes_neighbor_depth_features

        for transformer, cross_transformer in zip(self.transformers, self.cross_transformers):
            # self-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_features = transformer(src_node_features, torch.from_numpy(src_neighbor_times).to(src_node_features.device).to(torch.float32))
            # src_node_features = transformer(src_node_features)
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            dst_node_features = transformer(dst_node_features, torch.from_numpy(dst_neighbor_times).to(src_node_features.device).to(torch.float32))
            # dst_node_features = transformer(dst_node_features)

            # cross-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_embeddings = cross_transformer(inputs_query=src_node_features, inputs_key=dst_node_features,
                                              inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            dst_node_embeddings = cross_transformer(inputs_query=dst_node_features, inputs_key=src_node_features,
                                              inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)

            # # cross-attention block
            # # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            # src_node_embeddings = transformer(dst_node_features)
            # # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            # dst_node_embeddings = transformer(src_node_features)

            src_node_features, dst_node_features = src_node_embeddings, dst_node_embeddings

        # retrieve the embedding of the corresponding target node, which is at the first position of the sequence
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(torch.mean(src_node_embeddings, dim=1))
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(torch.mean(dst_node_embeddings, dim=1))

        return src_node_embeddings, dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge, time and depth features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_edge_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_neighbor_times: ndarray, shape (batch_size, num_neighbors + 1)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        assert nodes_neighbor_ids.shape[1] == self.depth_embedding.weight.shape[0]
        # Tensor, shape (num_neighbors + 1, node_feat_dim)
        nodes_neighbor_depth_features = self.depth_embedding(torch.tensor(range(nodes_neighbor_ids.shape[1])).to(self.device))

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_depth_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()