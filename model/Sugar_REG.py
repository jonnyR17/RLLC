
import os
import sys

from dgl import batch
from torch.ao.nn.quantized import Dropout
from typing import Optional

from torch.nn.functional import embedding

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn import GraphNorm

import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, GATConv, global_mean_pool, GINEConv,GINConv, global_max_pool, global_add_pool,MLP,GPSConv
from torch.nn import Linear, Sequential, ReLU, LayerNorm,Embedding ,BatchNorm1d as BN
from torch_scatter import scatter_mean
import torch.nn.functional as F
import torch.nn as nn

import os

class Net(torch.nn.Module):
    """
    net = Net(max_layer=2,
                node_dim=graphs[0].x.shape[1],
                hid_dim=args.hid_dim,
                out_dim=graphs.num_classes,
                sub_num=args.sub_num,
                sub_size=15,
                loss_type=0,
                sub_coeff=args.sub_coeff,
                mi_coeff=args.mi_coeff,
                device=torch.device('cuda'))
    """
    def __init__(self, max_layer, node_dim, hid_dim, out_dim, sub_num, sub_size, loss_type, sub_coeff, mi_coeff, device):
        super(Net, self).__init__()
        self.sub_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.device = device
        pe_dim = 8
        self.node_encoder = nn.Linear(11, hid_dim)
        self.node_emb = Embedding(19, hid_dim - pe_dim)
        self.node_emb1 = Embedding(19,hid_dim)
        self.conv = GINEConv(
            Sequential(
                Linear(hid_dim, hid_dim),
                ReLU(),
                Linear(hid_dim, hid_dim),
                ReLU(),
                BN(hid_dim),
            ), train_eps=False)
        self.conv2 = GINConv(
            Sequential(
                Linear(hid_dim, hid_dim),
                ReLU(),
                Linear(hid_dim, hid_dim),
                ReLU(),
                BN(hid_dim),
            ), train_eps=False)

        self.convs = torch.nn.ModuleList()
        for i in range(max_layer):#max_layer = 2
            self.convs.append(
                GINEConv(
                    Sequential(
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        BN(hid_dim),
                    ),train_eps=False))
        self.convs_s = torch.nn.ModuleList()
        for i in range(1):  # max_layer = 2
            self.convs_s.append(
                GINConv(
                    Sequential(
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        GraphNorm(hid_dim)
                    ), train_eps=False))
        self.lin1 = Linear(hid_dim, hid_dim)
        self.lin2 = Linear(hid_dim*2, hid_dim)
        self.lin3 = Linear(hid_dim*3, hid_dim)
        self.lin_s = Linear(hid_dim, hid_dim)
        self.edge_encoder = Embedding(4, hid_dim)
        self.pe_norm = BN(20)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.mlp = Sequential(
            Linear(hid_dim, hid_dim // 2),
            ReLU(),
            Linear(hid_dim // 2, hid_dim // 4),
            ReLU(),
            Linear(hid_dim // 4, 1),
        )
        self.mlp1 = Sequential(
            Linear(hid_dim, hid_dim // 2),
            ReLU(),
            Linear(hid_dim // 2, hid_dim // 4),
            ReLU(),
            Linear(hid_dim // 4, 1),
        )

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.discriminator = MLPDiscriminator(hid_dim)
        self.disc = Discriminator(hid_dim)
        self.b_xent = BCEWithLogitsLoss()
        self.redraw_projection = RedrawProjection(
            self,
            redraw_interval=1000 )
    def reset_parameters(self):
        self.conv2.reset_parameters()
        self.conv.reset_parameters()
        self.node_emb.reset_parameters()
        self.node_emb1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for conv in self.convs_s:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin_s.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_encoder.reset_parameters()


    def to(self, device):
        self.conv2.to(device)
        self.conv.to(device)

        for conv in self.convs:
            conv.to(device)
        for conv in self.convs_s:
            conv.to(device)
        self.lin1.to(device)
        self.lin2.to(device)
        self.lin3.to(device)
        self.edge_encoder.to(device)
        self.node_encoder.to(device)
        self.pe_norm.to(device)
        self.pe_lin.to(device)
        self.node_emb.to(device)
        self.node_emb1.to(device)
        self.mlp.to(device)
        self.mlp1.to(device)
        self.alpha.to(device)
        self.discriminator.to(device)
        self.lin_s.to(device)
        self.disc.to(device)

    def forward(self, data, agent, tau=0.4):
        # ===== 原图 GNN =====

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        pe = getattr(data, "pe", None)

        x_idx = x.argmax(dim=1).long()  # 转成整数索引
          # [num_nodes, emb_dim]
        if hasattr(data, "pe"):
            x_emb = self.node_emb(x_idx)
            x_pe = self.pe_norm(pe)
            x = torch.cat((x_emb, self.pe_lin(x_pe)),dim=1)
        else:
            x = self.node_emb1(x_idx)

        if edge_attr.dim() == 1:
            edge_attr = self.edge_encoder(edge_attr)
            data.edge_attr = edge_attr
        elif edge_attr.dim() == 2:
            num_cols = edge_attr.size(1)
            edge_attr_encoded = []
            for i in range(num_cols):
                num_classes = int(edge_attr[:, i].max().item()) + 1
                edge_attr_encoded.append(self.edge_encoder(edge_attr[:, i]))
            if len(edge_attr_encoded) > 1:
                edge_attr = torch.cat(edge_attr_encoded, dim=-1)
            else:
                edge_attr = edge_attr_encoded[0]
            edge_attr = self.lin3(edge_attr)

        batch = batch.to(self.device)
        for conv in self.convs:
            x = conv(x, edge_index,edge_attr)


        # ===== pool 子图 =====
        pool_x, sub_embedding, sub_labels= self.pool(x, data, agent)
        # pool_x, sub_embedding,sub_labels= self.pool_mlp(x, data, agent)#pool_x为草图节点表示，sub_embedding为草图的整图表示


        lbl = torch.cat([torch.ones_like(sub_labels), torch.zeros_like(sub_labels)], dim=0).float().to(self.device)
        logits = self.MI(pool_x, sub_embedding)
        loss_mi = self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))

        # 子图预测 loss
        sub_predict = self.mlp(torch.cat(sub_embedding,dim=0))
        loss_sub = F.l1_loss(sub_predict.view(-1), sub_labels.cuda())


        x = global_add_pool(x, batch)
        x = torch.cat([x, pool_x],dim=-1)
        # x = x+pool_x

        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        predicts = self.mlp(x).view(-1)

        loss_label = F.l1_loss(predicts, data.y.view(-1).float().to(self.device))
        loss = loss_label + self.sub_coeff * loss_sub + self.mi_coeff * loss_mi

        # 返回值保持原样
        return predicts, loss

    def MI(self, graph_embeddings, sub_embeddings):
        idx = torch.arange(graph_embeddings.shape[0]-1, -1, -1)
        idx[len(idx)//2] = idx[len(idx)//2+1]
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub))
            c_1_list.append(c_1.expand_as(sub))
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def regression_contrastive_loss(self, embeddings, targets, tau=0.4, temperature=0.2):
        """
        embeddings: [batch_size, hid_dim] 子图或图 embedding
        targets: [batch_size] 回归标签
        tau: 正样本阈值
        temperature: 对比损失温度参数
        """
        # 归一化
        z = F.normalize(embeddings, dim=-1)  # [B, d]

        # cosine 相似度矩阵
        sim_matrix = torch.mm(z, z.t())  # [B, B]
        sim_matrix = sim_matrix / temperature

        # targets 之间的差值矩阵
        y_diff = torch.abs(targets.unsqueeze(1) - targets.unsqueeze(0))  # [B, B]

        # 构建正负样本 mask
        pos_mask = (y_diff < tau).float()
        # 对角线视为正样本
        pos_mask.fill_diagonal_(1.0)

        # 对比 loss (InfoNCE 风格)
        log_prob = F.log_softmax(sim_matrix, dim=1)  # [B, B]
        loss = - (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)
        loss = loss.mean()
        return loss

    def pool_mlp(self, graph_node_embedding, data, agent):#graph_node_embedding = x,data= data,agent = Agent_chain
        xs, labels = [], []

        for index, graph in enumerate(data.to_data_list()):#to_data_list 将data从batch格式返回一个列表格式
            sub_embedding = graph_node_embedding[(data.batch == index).nonzero().flatten()]#通过index选择数据，提取非零数据并降维至一维
            sub_graph, edge_index,subg_node_index = agent.predict(sub_embedding.cpu(), graph.cpu(), index)#sub_graph 为子图集合，edge_index为子图边集合，subg_node_index为子图划分。
            sub_graph, edge_index = sub_graph.to(self.device), edge_index.to(self.device)

            #生成每个子图的表示
            x_sub = sub_graph.x.float()
            x_sub = self.conv2(x_sub, sub_graph.edge_index)
            x_sk = global_add_pool(x_sub, batch=sub_graph.batch)

            for conv in self.convs_s:
                x_sk = conv(x_sk, edge_index)
            x_sk = F.dropout(x_sk, p=0.5, training=self.training)
            xs.append(x_sk)
            labels.append(graph.y.expand(x_sk.shape[0]))

        out = torch.stack([x_sk.mean(0) for x_sk in xs])
        return out, xs,torch.cat(labels)


    def pool(self, graph_node_embedding, data, agent):#graph_node_embedding = x,data= data,agent = Agent_chain
        xs, labels = [], []

        for index, graph in enumerate(data.to_data_list()):#to_data_list 将data从batch格式返回一个列表格式
            sub_embedding = graph_node_embedding[(data.batch == index).nonzero().flatten()]#通过index选择数据，提取非零数据并降维至一维
            sub_graph, edge_index,subg_node_index = agent.predict(sub_embedding.cpu(), graph.cpu(), index)#sub_graph 为子图集合，edge_index为子图边集合，subg_node_index为子图划分。
            sub_graph, edge_index = sub_graph.to(self.device), edge_index.to(self.device)
            x_sub = sub_graph.x.float()
            for conv in self.convs_s:
                x_sub = conv(x_sub, sub_graph.edge_index)
            x_sk = F.dropout(x_sub, p=0.5, training=self.training)
            x_sk = global_mean_pool(x_sub, batch=sub_graph.batch)
            #生成每个子图的表示

            for conv in self.convs_s:
                x_sk = conv(x_sk, edge_index)
            x_sk = F.dropout(x_sk, p=0.5, training=self.training)
            xs.append(x_sk)
            labels.append(graph.y.view(-1).repeat(x_sk.shape[0]))

        out = torch.stack([x_sk.mean(0) for x_sk in xs])
        return out, xs, torch.cat(labels)

    def add_random_walk_pe(self, data, pe_dim=8, steps=4):
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # 构造邻接矩阵
        adj = torch.zeros((num_nodes, num_nodes), device=data.x.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        # 度矩阵倒数
        deg = adj.sum(dim=1, keepdim=True)
        deg = torch.where(deg > 0, deg, torch.ones_like(deg))  # 避免除0
        adj_norm = adj / deg  # 得到归一化转移矩阵

        # 初始化随机游走编码 (num_nodes x pe_dim)
        rw = torch.eye(num_nodes, device=data.x.device)

        # 进行 k 步随机游走
        for _ in range(steps):
            rw = torch.matmul(adj_norm, rw)

        # 只取前 pe_dim 维度
        if rw.shape[1] < pe_dim:
            rw = torch.cat([rw, torch.zeros(num_nodes, pe_dim - rw.shape[1], device=rw.device)], dim=1)
        else:
            rw = rw[:, :pe_dim]

        # 拼接到 x
        data.x = torch.cat([data.x, rw], dim=1)
        return data

    def pool_bak(self, big_graph_list, sketch_graph):
        xs = []
        for graph, edge_index in zip(big_graph_list, sketch_graph):
            x = self.conv2(graph.x.float(), graph.edge_index)
            x = global_mean_pool(x, batch=graph.batch)
            for conv in self.convs2:
                x = conv(x, edge_index)
            xs.append(x.mean(0))
        out = torch.stack(xs)
        return out

    def __repr__(self):
        return self.__class__.__name__

    def save(self, path):
        save_path = os.path.join(path, self.__class__.__name__+'.pt')
        torch.save(self.state_dict(), save_path)
        return save_path


import torch
from torch_scatter import scatter_add


def random_walk_return_prob(edge_index, num_nodes, steps=5, restart_prob=0.5, device=None):
    """
    计算随机游走返回概率矩阵。

    Args:
        edge_index: [2, E] 张量，PyG格式的边索引
        num_nodes: 节点数
        steps: 随机游走的步数
        restart_prob: 每一步回到起点的概率 (0 表示普通随机游走，>0 表示带重启)
        device: 计算设备

    Returns:
        return_prob: [num_nodes, num_nodes] 随机游走返回概率矩阵
    """
    if device is None:
        device = edge_index.device

    # 1. 构造稀疏邻接矩阵 (转移概率矩阵)
    row, col = edge_index
    # 计算度数
    deg = scatter_add(torch.ones_like(row, dtype=torch.float, device=device), row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv[deg == 0] = 0.0  # 避免除零

    # 概率矩阵 P (num_nodes x num_nodes)
    P = torch.sparse_coo_tensor(edge_index, deg_inv[row], (num_nodes, num_nodes), device=device)

    # 2. 初始化 return probability 矩阵
    R = torch.eye(num_nodes, device=device)  # 起点矩阵 I
    return_prob = torch.zeros_like(R)

    # 3. 迭代 steps 步
    for _ in range(steps):
        R = torch.sparse.mm(P, R)  # R = P * R
        return_prob += R * (1 - restart_prob)  # 累加概率
        # 如果带重启，则加回 I
        if restart_prob > 0:
            R = restart_prob * torch.eye(num_nodes, device=device) + (1 - restart_prob) * R

    return return_prob


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c: 1, 512; h_pl: 1, 2708, 512; h_mi: 1, 2708, 512
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)

        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits

class MLPDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_sub, h_graph):
        # 拼接子图和图表示
        h = torch.cat([h_sub, h_graph], dim=-1)  # [num_subgraphs, hidden_dim*2]
        return self.net(h).squeeze(-1)  # [num_subgraphs]


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1

def map_subgraph_to_node(subgraph_emb, subgraph_nodes, num_nodes):
    """
    subgraph_emb: [num_subgraphs, hid_dim] tensor
    subgraph_nodes: List[List[int]], 每个子图对应的节点索引
    num_nodes: 原图节点总数
    """
    device = subgraph_emb.device
    hid_dim = subgraph_emb.size(1)

    # 初始化累加表示和计数
    node_emb_sum = torch.zeros(num_nodes, hid_dim, device=device)
    node_count = torch.zeros(num_nodes, device=device)

    # 遍历子图，将子图表示累加到每个节点
    for sg_idx, nodes in enumerate(subgraph_nodes):
        emb = subgraph_emb[sg_idx]  # [hid_dim]
        for n in nodes:
            node_emb_sum[n] += emb
            node_count[n] += 1

    # 避免除零
    node_emb = torch.where(
        node_count.unsqueeze(1) > 0,
        node_emb_sum / node_count.unsqueeze(1),
        torch.zeros_like(node_emb_sum)
    )
    return node_emb
