
import os
import sys
BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, GATConv, global_mean_pool,GINEConv, GINConv, global_max_pool, global_add_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN,Embedding
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
        pe_dim = 8
        self.pe_norm = BN(20)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.mi_coeff = mi_coeff
        self.device = device
        self.conv1 = GINEConv(
            Sequential(
                Linear(128, hid_dim),
                ReLU(),
                Linear(hid_dim, hid_dim),
                ReLU(),
                BN(hid_dim),
            ), train_eps=False)
        self.conv2 = GATConv(
            in_channels=hid_dim,
            out_channels=hid_dim,
            heads=1,
            concat=False
        )
        self.conv3 = GATConv(
            in_channels=hid_dim,
            out_channels=hid_dim,
            heads=1,
            concat=False
        )
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
                    ), train_eps=False))
        self.convs1 = torch.nn.ModuleList()
        for i in range(max_layer):  # max_layer = 2
            self.convs1.append(
                GINEConv(
                    Sequential(
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        Linear(hid_dim, hid_dim),
                        ReLU(),
                        BN(hid_dim),
                    ), train_eps=False))
        self.convs2 = torch.nn.ModuleList()
        for i in range(max_layer - 1):
            self.convs2.append(HybridGATLayer(hid_dim)
                )
        self.convs3 = torch.nn.ModuleList()
        for i in range(max_layer - 1):
            self.convs3.append(
                HybridGATLayer(hid_dim))
        self.lin1 = Linear(hid_dim*2, hid_dim)
        self.lin2 = Linear(hid_dim, out_dim)
        self.disc = Discriminator(hid_dim)
        self.b_xent = BCEWithLogitsLoss()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.node_emb = Embedding(node_dim, hid_dim - pe_dim)
        self.node_emb1 = Embedding(node_dim, hid_dim)
        # For continuous node features (e.g., PROTEINS), use linear projections instead of argmax+Embedding
        self.x_lin_pe = nn.Linear(node_dim, hid_dim - pe_dim)
        self.x_lin = nn.Linear(node_dim, hid_dim)

        self.edge_encoder = Embedding(4, hid_dim)

    # def reset_parameters(self):
    #     self.conv1.reset_parameters()
    #     self.convs1.reset_parameters()
    #     self.conv2.reset_parameters()
    #     self.conv3.reset_parameters()
    #     for conv in self.convs:
    #         conv.reset_parameters()
    #     for conv in self.convs2:
    #         conv.reset_parameters()
    #     self.lin1.reset_parameters()
    #     self.lin2.reset_parameters()
    #     self.node_emb.reset_parameters()
    #     self.node_emb1.reset_parameters()
    #     self.pe_lin.reset_parameters()
    def reset_parameters(self):
        # Standard parameter reset
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for conv in self.convs1:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for conv in self.convs2:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for conv in self.convs3:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        if hasattr(self.disc, "f_k") and hasattr(self.disc.f_k, "reset_parameters"):
            self.disc.f_k.reset_parameters()
        if hasattr(self.pe_norm, "reset_parameters"):
            self.pe_norm.reset_parameters()
        self.pe_lin.reset_parameters()

        self.node_emb.reset_parameters()
        self.node_emb1.reset_parameters()
        self.x_lin_pe.reset_parameters()
        self.x_lin.reset_parameters()
        self.edge_encoder.reset_parameters()

        with torch.no_grad():
            self.alpha.fill_(1.0)

    # def to(self, device):
    #     self.conv1.to(device)
    #     self.convs1.to(device)
    #     self.conv2.to(device)
    #     self.conv3.to(device)
    #     for conv in self.convs:
    #         conv.to(device)
    #     for conv in self.convs2:
    #         conv.to(device)
    #     for conv in self.convs3:
    #         conv.to(device)
    #     self.lin1.to(device)
    #     self.lin2.to(device)
    #     self.disc.to(device)
    #     self.alpha.to(device)
    #     self.pe_norm.to(device)
    #     self.pe_lin.to(device)
    #     self.node_emb.to(device)
    #     self.node_emb1.to(device)
    #     self.edge_encoder.to(device)
    def to(self, device):
        # Keep an explicit device attribute for parts of the code that reference self.device
        self.device = device
        return super().to(device)

    @staticmethod
    def _looks_one_hot(x: torch.Tensor, atol: float = 1e-4) -> bool:
        """Heuristic: detect one-hot-like float features."""
        if x is None or x.dim() != 2 or x.size(1) < 2 or (not x.dtype.is_floating_point):
            return False
        # Sample rows to avoid huge overhead on big graphs
        n = x.size(0)
        if n > 2048:
            idx = torch.randperm(n, device=x.device)[:2048]
            xs = x[idx]
        else:
            xs = x
        if xs.min().item() < -atol or xs.max().item() > 1.0 + atol:
            return False
        row_sum = xs.sum(dim=1)
        return torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-2)

    def _encode_node_x(self, x: torch.Tensor, with_pe: bool) -> torch.Tensor:
        """Encode node features: one-hot/int -> Embedding; continuous -> Linear."""
        if x is None:
            raise ValueError("data.x is None (no node features). Please add a transform to create x.")
        if x.dtype in (torch.long, torch.int64, torch.int32):
            idx = x.view(-1).long()
            return self.node_emb(idx) if with_pe else self.node_emb1(idx)
        if self._looks_one_hot(x):
            idx = x.argmax(dim=1).long()
            return self.node_emb(idx) if with_pe else self.node_emb1(idx)
        # continuous
        return self.x_lin_pe(x.float()) if with_pe else self.x_lin(x.float())

    def _encode_edge_attr(self, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode edge attributes into [E, hid_dim]. If missing, return zeros."""
        device = edge_index.device
        E = edge_index.size(1)
        hid = self.edge_encoder.embedding_dim
        if edge_attr is None:
            return torch.zeros((E, hid), device=device)

        if edge_attr.dtype in (torch.long, torch.int64, torch.int32):
            idx = edge_attr.view(-1).long()
        elif edge_attr.dim() == 2 and edge_attr.size(1) > 1:
            idx = edge_attr.argmax(dim=1).long()
        else:
            # Continuous / unsupported edge attr: fall back to zeros to avoid crashes
            return torch.zeros((E, hid), device=device)

        idx = idx.clamp(0, self.edge_encoder.num_embeddings - 1).to(device)
        return self.edge_encoder(idx)

    # def forward(self, data, agent):
    #     # GNN
    #     if hasattr(data, 'pe'):
    #         x, edge_index, edge_attr, batch , pe = data.x, data.edge_index, data.edge_attr, data.batch, data.pe
    #         x_pe = self.pe_norm(pe)
    #         x_idx = x.argmax(dim=1).long()  # 转成整数索引
    #         x_emb = self.node_emb(x_idx)
    #         x = torch.cat((x_emb, self.pe_lin(x_pe)), dim=1)
    #     else:
    #         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    #         x = x.argmax(dim=1).long()  # 转成整数索引
    #         x = self.node_emb1(x)
    #
    #
    #     if edge_attr is not None:
    #         edge_attr = edge_attr.argmax(dim=1).long()
    #         edge_attr = self.edge_encoder(edge_attr)
    #         data.edge_attr = edge_attr
    #
    #         for conv in self.convs1:
    #             x = conv(x, edge_index,edge_attr)
    #     else:
    #         for conv in self.convs1:
    #             x = conv(x, edge_index)
    def forward(self, data, agent):
        # Use the actual module device (avoid hard-coded cuda:0)
        device = next(self.parameters()).device
        self.device = device

        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        # ---- Node feature encoding ----
        if hasattr(data, 'pe'):
            pe = data.pe.to(device)
            x_pe = self.pe_norm(pe)
            x_node = self._encode_node_x(data.x.to(device), with_pe=True)
            x = torch.cat((x_node, self.pe_lin(x_pe)), dim=1)
        else:
            x = self._encode_node_x(data.x.to(device), with_pe=False)

        # ---- Edge feature encoding ----
        edge_attr = getattr(data, 'edge_attr', None)
        if isinstance(edge_attr, torch.Tensor):
            edge_attr = edge_attr.to(device)
        edge_emb = self._encode_edge_attr(edge_attr, edge_index)

        # ---- GNN backbone ----
        for conv in self.convs1:
            x = conv(x, edge_index, edge_emb)

            # RL

        # RL
        pool_x, sub_embedding, sub_labels= self.pool(x, data, agent)

        # loss_MI
        lbl = torch.cat([torch.ones_like(sub_labels), torch.zeros_like(sub_labels)], dim=0).float().to(self.device)
        logits = self.MI(pool_x, sub_embedding)
        loss_mi = self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))

        # loss_sub
        sub_predict = F.log_softmax(self.lin2(torch.cat(sub_embedding, dim=0)), dim=-1)
        loss_sub = F.nll_loss(sub_predict, sub_labels.to(device))

        x_g = pool_x[data.batch]
        x = torch.cat([x,x_g],dim=-1)
        x = global_mean_pool(x, data.batch)
        # x = torch.cat((global_mean_pool(x, data.batch), pool_x), dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        predicts = F.log_softmax(x, dim=-1)
        loss_label = F.nll_loss(predicts, data.y.view(-1).to(device))

        loss = loss_label + self.sub_coeff * loss_sub + self.mi_coeff * loss_mi

        return predicts, loss
    
    # def MI(self, graph_embeddings, sub_embeddings):
    #     idx = torch.arange(graph_embeddings.shape[0]-1, -1, -1)
    #     idx[len(idx)//2] = idx[len(idx)//2+1]
    #     shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
    #     c_0_list, c_1_list = [], []
    #     for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
    #         c_0_list.append(c_0.expand_as(sub))
    #         c_1_list.append(c_1.expand_as(sub))
    #     c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
    #     return self.disc(sub, c_0, c_1)
    def MI(self, graph_embeddings, sub_embeddings):
        """Mutual-information-style discriminator: (sub, graph) vs (sub, shuffled_graph)."""
        n = graph_embeddings.size(0)
        if n <= 1:
            shuffle_embeddings = graph_embeddings
        elif n == 2:
            shuffle_embeddings = graph_embeddings.flip(0)
        else:
            perm = torch.randperm(n, device=graph_embeddings.device)
            # avoid identity permutation
            if torch.all(perm == torch.arange(n, device=perm.device)):
                perm = torch.roll(perm, 1)
            shuffle_embeddings = graph_embeddings[perm]

        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub))
            c_1_list.append(c_1.expand_as(sub))
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    @torch.no_grad()
    def encode_nodes(self, data):
        """
        返回经过 convs1 后的节点嵌入 x（与 data.x 的节点顺序一致）
        用于预训练智能体，不走 pool / MI / label loss。
        """
        if hasattr(data, 'pe'):
            x, edge_index, edge_attr, pe = data.x, data.edge_index, data.edge_attr, data.pe
            x_pe = self.pe_norm(pe)
            x_idx = x.argmax(dim=1).long()
            x_emb = self.node_emb(x_idx)
            x = torch.cat((x_emb, self.pe_lin(x_pe)), dim=1)
        else:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            x = x.argmax(dim=1).long()
            x = self.node_emb1(x)

        if edge_attr is not None:
            edge_attr = edge_attr.argmax(dim=1).long()
            edge_attr = self.edge_encoder(edge_attr)
            data.edge_attr = edge_attr
            for conv in self.convs1:
                x = conv(x, edge_index, edge_attr)
        else:
            for conv in self.convs1:
                x = conv(x, edge_index)

        return x
    def pool(self, graph_node_embedding, data, agent):#graph_node_embedding = x,data= data,agent = Agent_chain
        xs, labels = [], []
        for index, graph in enumerate(data.to_data_list()):#to_data_list 将data从batch格式返回一个列表格式
            sub_embedding = graph_node_embedding[(data.batch == index).nonzero().flatten()]#通过index选择数据，提取非零数据并降维至一维

            # sub_graph, edge_index, _ = leiden_cut_to_subgraphs(
            #     pyg_graph=graph.cpu(),
            #     node_feat=sub_embedding.detach().cpu(),
            #     resolution=1,
            #     seed=1,
            #     make_undirected=True,
            #     keep_topk=None,  # 或按需求设置
            #     min_size=0
            # )

            sub_graph, edge_index,sub_node_embedding= agent.predict(sub_embedding.cpu(), graph.cpu(), index)#agent_chain().predict
            sub_graph, edge_index = sub_graph.to(self.device), edge_index.to(self.device)

            # if sub_graph.size(0) >1:
            if getattr(sub_graph, 'num_graphs', 1) > 1:
                x = self.conv2(sub_graph.x.float(), sub_graph.edge_index)
            else:
                x = self.conv3(sub_graph.x.float(), sub_graph.edge_index)
            x = global_mean_pool(x, batch=sub_graph.batch)
            if x.size(0) >1:
                for conv in self.convs2:
                    x = conv(x, edge_index)
            else:
                for conv in self.convs3:
                    x = conv(x,edge_index)
            x = F.dropout(x, p=0.5, training=self.training)
            xs.append(x)
            labels.append(graph.y.expand(x.shape[0]))

        out = torch.stack([x.mean(0) for x in xs])
        return out, xs, torch.cat(labels)

    #k-hop subgraph
    # def pool(self, graph_node_embedding, data, agent):#graph_node_embedding = x,data= data,agent = Agent_chain
    #     xs, labels,node_embedings = [], [],[]
    #     for index, graph in enumerate(data.to_data_list()):#to_data_list 将data从batch格式返回一个列表格式
    #         sub_embedding = graph_node_embedding[(data.batch == index).nonzero().flatten()]#通过index选择数据，提取非零数据并降维至一维
    #         nodes = agent.predict(sub_embedding.cpu(), graph.cpu(), index)#agent_chain().predict
    #         # sub_graph, edge_index = sub_graph.to(self.device), edge_index.to(self.device)
    #         for candidate in nodes:
    #             candidate = int(candidate)
    #             subset,edge_index,_,_ =k_hop_subgraph(candidate,3,data.edge_index)
    #             node_embeding = graph_node_embedding[subset].mean(dim=0)
    #             node_embedings.append(node_embeding)
    #         x = torch.stack(node_embedings)
    #
    #         xs.append(x)
    #         labels.append(graph.y.expand(x.shape[0]))
    #     out = torch.stack([x.mean(0) for x in xs])
    #     return out, xs, torch.cat(labels)

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
from torch.nn import LayerNorm
from torch_geometric.nn import GATv2Conv

class HybridGATLayer(torch.nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.mlp = Sequential(
            Linear(hid_dim, hid_dim),
            ReLU(),
            Linear(hid_dim, hid_dim),
            ReLU(),
            LayerNorm(hid_dim),
        )
        self.conv = GATConv(hid_dim, hid_dim, heads=4, concat=False, dropout=0)

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = self.conv(x, edge_index)
        return x


import torch
import igraph as ig
import leidenalg
from typing import Optional, List

from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, to_undirected, coalesce


def leiden_cut_to_subgraphs(
    pyg_graph: Data,
    node_feat: torch.Tensor,
    resolution: float = 1.0,
    seed: int = 42,
    make_undirected: bool = True,
    keep_topk: Optional[int] = None,
    min_size: int = 2,
):
    """
    使用 Leiden 代替 Louvain 做社区划分。
    输入:
      - pyg_graph: 单张 PyG Data
      - node_feat: [num_nodes, d] 节点特征（通常是 sub_embedding）
    输出:
      - sub_graph_batch: Batch
      - comm_edge_index: 社区级图 edge_index
      - comm_id: [num_nodes] 每个节点所属社区 id；未被保留为 -1
    """
    edge_index = pyg_graph.edge_index
    num_nodes = pyg_graph.num_nodes

    if num_nodes is None or num_nodes == 0:
        empty = Batch.from_data_list([
            Data(x=node_feat.new_zeros((0, node_feat.size(-1))),
                 edge_index=edge_index.new_zeros((2, 0)))
        ])
        return empty, edge_index.new_zeros((2, 0)), torch.full((0,), -1, dtype=torch.long)

    if node_feat.size(0) != num_nodes:
        raise ValueError("node_feat.size(0) must equal num_nodes")

    if make_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    # --- 1) PyG -> igraph ---
    ei = edge_index.detach().cpu()
    edges = ei.t().tolist()

    # igraph 需要 List of tuples
    g = ig.Graph(n=num_nodes, edges=edges, directed=False)
    if g.ecount() == 0:
        # 没边的图直接退化
        communities = [[i] for i in range(num_nodes)]
    else:
        # --- 2) Leiden 算法 ---
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=seed,
        )
        # part 是一个 VertexClustering，对象

        communities: List[List[int]] = [list(c) for c in part]
        communities.sort(key=len, reverse=True)

    # --- 3) 过滤 ---
    if keep_topk is not None:
        communities = communities[:keep_topk]
    if min_size > 1:
        communities = [c for c in communities if len(c) >= min_size]

    if len(communities) == 0:
        communities = [list(range(num_nodes))]

    # --- 4) 构造 comm_id ---
    comm_id = torch.full((num_nodes,), -1, dtype=torch.long)
    for new_cid, nodes in enumerate(communities):
        comm_id[torch.tensor(nodes, dtype=torch.long)] = new_cid
    n_comm = len(communities)

    # --- 5) 社区子图 Batch ---
    subgraphs = []
    for nodes in communities:
        nodes_t = torch.tensor(nodes, dtype=torch.long)
        sub_ei, _ = subgraph(nodes_t, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        sub_x = node_feat[nodes_t]
        subgraphs.append(Data(x=sub_x, edge_index=sub_ei))
    sub_graph_batch = Batch.from_data_list(subgraphs)

    # --- 6) 社区级图 ---
    src = comm_id[edge_index[0].cpu()]
    dst = comm_id[edge_index[1].cpu()]

    mask = (src >= 0) & (dst >= 0) & (src != dst)
    if mask.any():
        comm_ei = torch.stack([src[mask], dst[mask]], dim=0).long()
        comm_ei = to_undirected(comm_ei, num_nodes=n_comm)
        comm_ei, _ = coalesce(comm_ei, None, n_comm, n_comm)
    else:
        comm_ei = torch.empty((2, 0), dtype=torch.long)

    return sub_graph_batch, comm_ei, comm_id

