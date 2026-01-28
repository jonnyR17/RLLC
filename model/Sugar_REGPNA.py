import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BCEWithLogitsLoss
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, global_mean_pool
from torch_geometric.utils import degree


class Net(torch.nn.Module):
    def __init__(self, max_layer, node_dim, hid_dim, out_dim, sub_num, sub_size,
                 loss_type, sub_coeff, mi_coeff, device):
        super(Net, self).__init__()
        self.sub_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.device = device
        self.node_encoder = nn.Linear(9, hid_dim)
        self.edge_encoder = nn.Linear(4, hid_dim)

        # PNAConv 参数
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        # 原图卷积层
        self.conv2 = PNAConv(hid_dim, hid_dim, aggregators=aggregators,
                             scalers=scalers, deg=torch.ones(10), edge_dim=hid_dim, towers=4, pre_layers=1, post_layers=1)
        self.conv3 = PNAConv(hid_dim, hid_dim, aggregators=aggregators,
                             scalers=scalers, deg=torch.ones(10), edge_dim=hid_dim, towers=4, pre_layers=1, post_layers=1)
        self.convs = nn.ModuleList([
            PNAConv(hid_dim, hid_dim, aggregators=aggregators, scalers=scalers,
                    deg=torch.ones(10), edge_dim=hid_dim, towers=4, pre_layers=1, post_layers=1)
            for _ in range(max_layer - 1)
        ])

        # 子图卷积层
        self.convs2 = nn.ModuleList([
            PNAConv(hid_dim, hid_dim, aggregators=aggregators, scalers=scalers,
                    deg=torch.ones(10), edge_dim=hid_dim, towers=4, pre_layers=1, post_layers=1)
            for _ in range(max_layer - 1)
        ])
        self.convs3 = nn.ModuleList([
            PNAConv(hid_dim, hid_dim, aggregators=aggregators, scalers=scalers,
                    deg=torch.ones(10), edge_dim=hid_dim, towers=4, pre_layers=1, post_layers=1)
            for _ in range(max_layer - 1)
        ])

        # BatchNorm
        self.bn2 = BatchNorm(hid_dim)
        self.bn3 = BatchNorm(hid_dim)
        self.bns = nn.ModuleList([BatchNorm(hid_dim) for _ in range(max_layer - 1)])
        self.bns2 = nn.ModuleList([BatchNorm(hid_dim) for _ in range(max_layer - 1)])
        self.bns3 = nn.ModuleList([BatchNorm(hid_dim) for _ in range(max_layer - 1)])

        # 线性层
        self.lin1 = Linear(hid_dim, hid_dim)
        self.lin2 = Linear(hid_dim, 1)
        self.lin3 = Linear(hid_dim, 1)
        self.disc = Discriminator(hid_dim)
        self.b_xent = BCEWithLogitsLoss()

    def forward(self, data, agent, tau=0.1):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x.float())
        edge_attr = F.one_hot(edge_attr, num_classes=4).float()
        edge_attr = self.edge_encoder(edge_attr)

        deg = degree(edge_index[0], num_nodes=data.num_nodes)

        # 原图卷积
        for i, conv in enumerate(self.convs):
            h_in = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(self.bns[i](x))
            x = F.dropout(x, p=0.3, training=self.training)
            x = x + h_in  # 残差

        # pool 子图
        # pool_x, sub_embedding, sub_labels = self.pool(x, data, agent)

        # sub_labels = sub_labels.view(-1).float().to(self.device)
        # sub_predict = self.lin2(pool_x).view(-1)
        # loss_sub = F.smooth_l1_loss(sub_predict, sub_labels, beta=0.5)

        x_cat = global_add_pool(x, data.batch)
        x_cat = F.relu(self.lin1(x_cat))
        x_cat = F.dropout(x_cat, p=0.5, training=self.training)
        predicts = self.lin3(x_cat).view(-1)
        loss_label = F.smooth_l1_loss(predicts, data.y.view(-1).float().to(self.device), beta=0.5)

        # loss_cl = self.regression_contrastive_loss(pool_x, data.y.view(-1).float().to(self.device), tau=tau, temperature=0.2)

        loss = loss_label

        return loss_label, loss

    def regression_contrastive_loss(self, embeddings, targets, tau=0.1, temperature=0.2):
        z = F.normalize(embeddings, dim=-1)
        sim_matrix = torch.mm(z, z.t()) / temperature
        y_diff = torch.abs(targets.unsqueeze(1) - targets.unsqueeze(0))
        pos_mask = (y_diff < tau).float()
        pos_mask.fill_diagonal_(1.0)
        log_prob = F.log_softmax(sim_matrix, dim=1)
        loss = - (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)
        return loss.mean()

    # def pool(self, graph_node_embedding, data, agent):
    #     xs, labels = [], []
    #     for index, graph in enumerate(data.to_data_list()):
    #         sub_embedding = graph_node_embedding[(data.batch == index).nonzero().flatten()]
    #         sub_graph, edge_index = agent.predict(sub_embedding.cpu(), graph.cpu(), index)
    #         sub_graph, edge_index = sub_graph.to(self.device), edge_index.to(self.device)
    #
    #         deg = degree(sub_graph.edge_index[0], num_nodes=sub_graph.num_nodes)
    #
    #         if sub_graph.size(0) > 1:
    #             x = self.conv2(sub_graph.x.float(), sub_graph.edge_index)
    #             x = F.relu(self.bn2(x))
    #         else:
    #             x = self.conv3(sub_graph.x.float(), sub_graph.edge_index)
    #             x = F.relu(self.bn3(x))
    #
    #         x = global_mean_pool(x, batch=sub_graph.batch)
    #
    #         if x.size(0) > 1:
    #             h = x
    #             for i, conv in enumerate(self.convs2):
    #                 x = conv(x, edge_index)
    #                 x = F.relu(self.bns2[i](x))
    #                 x = x + h
    #         else:
    #             h = x
    #             for i, conv in enumerate(self.convs3):
    #                 x = conv(x, edge_index)
    #                 x = F.relu(self.bns3[i](x))
    #                 x = x + h
    #
    #         xs.append(x)
    #         labels.append(graph.y)
    #
    #     out = torch.stack([x.mean(0) for x in xs])
    #     return out, xs, torch.cat(labels)

    def add_random_walk_pe(self, data, pe_dim=8, steps=4):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        adj = torch.zeros((num_nodes, num_nodes), device=data.x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1, keepdim=True)
        deg = torch.where(deg > 0, deg, torch.ones_like(deg))
        adj_norm = adj / deg
        rw = torch.eye(num_nodes, device=data.x.device)
        for _ in range(steps):
            rw = torch.matmul(adj_norm, rw)
        if rw.shape[1] < pe_dim:
            rw = torch.cat([rw, torch.zeros(num_nodes, pe_dim - rw.shape[1], device=rw.device)], dim=1)
        else:
            rw = rw[:, :pe_dim]
        data.x = torch.cat([data.x, rw], dim=1)
        return data

    def reset_parameters(self):
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        for conv in self.convs + self.convs2 + self.convs3:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_encoder.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        sc_1 = self.f_k(h_pl, c)
        sc_2 = self.f_k(h_mi, c)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        return logits
