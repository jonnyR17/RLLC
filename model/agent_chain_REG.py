import sys

import numpy as np
import re
import time
import os
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from torch_geometric.utils import *
from torch_geometric.data import Batch
import torch_geometric as pyg
import tqdm
from collections import namedtuple
from itertools import product
from copy import deepcopy



class AgentChain(object):
    def __init__(self, update_target_estimator_every, time_step, max_k_hop, epochs, visual=False, ablation_depth=0, wandb=None):
        self.update_target_estimator_every = update_target_estimator_every
        self.time_step = time_step
        self.max_k_hop = max_k_hop
        self.train_epoches = 0
        self.epochs = epochs
        self.visual = visual
        self.cnt, self.max = 0, 2000
        # agent act normal when ablation_depth is falsel; act randomly when ablation_depth == -1; act fixed when ablation_depth == 2
        self.ablation = ablation_depth
        self.wandb = wandb
        self.callcunt=0

    def bind_selector(self, candidate):
        self.node_selector = candidate
        self.target_node_selector_net = self.snapshot()
        return self
    
    def load_best(self ,candidate):
        print(type(candidate))
        self.node_selector.qnet.load_state_dict(candidate.state_dict())
        # self.node_selector.qnet.load_state_dict(candidate)
        return self

    def snapshot(self):
        return deepcopy(self.node_selector.qnet)

    def predict(self, graph_node_embedding, graph, batch):
        """

        Args:
            graph_node_embedding: 两次卷积后的数据特征，格式为一维列表
            graph: data中的图，按照dataloder的顺寻排列，
            batch: index，所选图的索引号

        Returns:
            sub_graph:子图特征
            edge_index:子图边索引
        """
        self.callcunt+=1
        sub_data = []
        subgraph_index_list = []
        com = False
        node_list = remove_isolated_nodes(graph.edge_index)[-1].nonzero().flatten().numpy().tolist()#去除无连边节点
        np.random.seed(int(time.time()))
        # self.communites = self.extract(graph)
        # self.communites = graph.subgraph_nodes

        self.candidate_node_list, self.communites = self.node_selector.predict(node_list, graph_node_embedding, graph, batch)

        # print(len(self.communites))
        if self.communites ==[]:
            print("bad choice graph indx:",graph)
            self.communites.append(node_list)
        if com:
            for community in self.communites:
                neighbors = torch.tensor(list(map(int,community)))
                if len(neighbors) == 1:
                    node = neighbors.item()
                    src, dst = graph.edge_index
                    one_hop_neighbors = torch.cat([
                        dst[src == node],
                        src[dst == node]
                    ]).unique()
                    neighbors = torch.unique(torch.cat([neighbors, one_hop_neighbors]))
                x = graph_node_embedding[neighbors]
                edge_index, edge_attr = subgraph(neighbors, graph.edge_index, edge_attr=graph.edge_attr,
                                                 relabel_nodes=True, num_nodes=graph.num_nodes)
                sub_data.append(pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_index=neighbors))
                subgraph_index_list.append(neighbors)
        else:  # 使用 k-hop ego 子图，以 candidate_node_list 作为根节点
            k = 2  # 例如 k-hop
            if self.candidate_node_list.numel() == 0:  # 等价于 == []
                num_nodes = graph.num_nodes
                n = min(5, num_nodes)  # 避免 n > 节点总数
                # 随机选择 n 个节点
                self.candidate_node_list = torch.randperm(num_nodes)[:n].tolist()
            for node in self.candidate_node_list:
                node = int(node)  # 确保为 int

                # 获取 k-hop 子图节点及边索引
                subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
                    node_idx=node,
                    num_hops=k,
                    edge_index=graph.edge_index,
                    relabel_nodes=True,
                    num_nodes=graph.num_nodes
                )

                # 节点 embedding
                x = graph_node_embedding[subset]

                # 对应的 edge_attr
                if graph.edge_attr is not None:
                    edge_attr = graph.edge_attr[edge_mask]
                else:
                    edge_attr = None

                # 构建子图 PyG Data
                sub_data.append(pyg.data.Data(
                    x=x,
                    edge_index=edge_index_sub,
                    edge_attr=edge_attr,
                    node_index=subset
                ))
                subgraph_index_list.append(subset)
        sub_data = Batch.from_data_list(sub_data)

        subgraph_index_list = [t.tolist() for t in  subgraph_index_list]
        return sub_data,self.gen_sketch_graph(subgraph_index_list,graph),subgraph_index_list

    def relabel_nodes(self,edge_index, nodes):
        node_dict = {node: i for i, node in enumerate(nodes)}
        src, dst = edge_index[0], edge_index[1]
        relabeled_src = torch.tensor([node_dict.get(node.item(), -1) for node in src])
        relabeled_dst = torch.tensor([node_dict.get(node.item(), -1) for node in dst])
        
        # 过滤掉不存在于节点字典中的边
        mask = (relabeled_src != -1) & (relabeled_dst != -1)
        relabeled_edge_index = torch.stack([relabeled_src[mask], relabeled_dst[mask]], dim=0)
        
        return relabeled_edge_index
    def merge_subgraphs(self,sub_data):
        # 合并所有子图的节点特征
        all_x = torch.cat([data.x for data in sub_data], dim=0)
        
        # 合并所有子图的边，并调整边的索引
        all_edge_index = []
        node_offset = 0
        
        for data in sub_data:
            # 调整边的索引
            edge_index = data.edge_index + node_offset
            all_edge_index.append(edge_index)
             
             # 更新节点偏移量
            node_offset += data.x.size(0)
            
            # 将所有边索引拼接在一起
        all_edge_index = torch.cat(all_edge_index, dim=1)
        
        return pyg.data.Data(x=all_x, edge_index=all_edge_index)
    def predict_fake(self, graph_node_embedding, graph, batch, ablation_depth=2):
        node_list = remove_isolated_nodes(graph.edge_index)[-1].nonzero().flatten().numpy().tolist()
        time_step = self.time_step if self.time_step < len(node_list) else len(node_list)
        np.random.seed(1234)
        self.candidate_node_list = self.node_selector(node_list,graph_node_embedding)

        # self.candidate_node_list = np.random.choice(node_list, time_step, replace=False)
        if ablation_depth == 2:
            self.depth_list = np.ones(len(self.candidate_node_list), dtype=np.int) * int(ablation_depth)
        elif ablation_depth == -1:
            self.depth_list = [np.random.choice(range(1, self.max_k_hop+1)) for _ in range(len(self.candidate_node_list))]
        sub_data, subgraph_index_list =  [], []
        for node, depth in zip(self.candidate_node_list, self.depth_list):
            node_index, edge_index, node_map, _ = k_hop_subgraph(int(node), int(depth), graph.edge_index, relabel_nodes=True)
            sub_data.append(pyg.data.Data(x=graph_node_embedding[node_index], edge_index=edge_index))
            subgraph_index_list.append(node_index.numpy().tolist())
        sub_data = Batch.from_data_list(sub_data)
        return sub_data, self.gen_sketch_graph(subgraph_index_list)

    def plot_sub_graph(self, sub_data, batch_index, subgraph_index_list):
        if not os.path.exists(f"./temp_sub/{batch_index}"):
            os.makedirs(f'./temp_sub/{batch_index}')

        for index, graph in enumerate(sub_data.to_data_list()):
            G = nx.Graph()
            for i,j in graph.edge_index.T:
                G.add_edge(int(i), int(j))
            nx.draw(G)
            plt.savefig(f"./temp_sub/{batch_index}/{index}.png")
            print(f"./temp_sub/{batch_index}/{index}.png saved!!!")
            plt.close()

    def gen_sketch_graph(self, subgraph_index_list, graph, eps=1):
        sub_num = len(subgraph_index_list)
        adj = torch.zeros((sub_num, sub_num))
        for i, j in product(range(sub_num), range(sub_num)):
            if self.has_edges_between(graph, subgraph_index_list[i], subgraph_index_list[j]):
                adj[i, j] = adj[j, i] = 1
            if i > j:
                continue
        return dense_to_sparse(adj)[0]

    def has_edges_between(self, graph, subgraph1, subgraph2):
        """检查两个子图的节点在原图中是否有直接连接"""
        edge_set = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
        return any((u in subgraph1 and v in subgraph2) or (v in subgraph1 and u in subgraph2)
                   for u, v in edge_set)
    

    def fed_reward_node(self, reward):
        self.node_selector.memory.fed_reward(reward)

    def train(self):
        for train_epochs in range(self.epochs):
            if train_epochs%self.update_target_estimator_every == 0:
                self.target_node_selector_net = self.snapshot()
            # depth_loss = self.depth_selector.train(self.target_depth_selector_net)
            # neighbor_loss = self.neighbor_selector.train(self.target_neighbor_selector_net)
            node_loss = self.node_selector.train(self.target_node_selector_net)
        self.clear()
        return  node_loss
    
    def _eval(self):

        self.node_selector._train = False

    def _train(self):#设置当前三个选择模块的训练模式指示器为真

        self.node_selector._train = True

    # def is_full(self):
    #     return self.depth_selector.is_full() and self.neighbor_selector.is_full()
    def is_full(self):
        return self.node_selector.is_full()
    
    def clear(self):
        # self.neighbor_selector.clear()
        # self.depth_selector.clear()
        self.node_selector.clear()

    def Cal_Q(self,partition, G):
        m = len(G.edges(None, False))
        a = []
        e = []
        for i in range(len(partition)):
            partition[i] = list(partition[i])
        for community in partition:
            t = 0.0
            for node in community:
                t += len([x for x in G.neighbors(node)])
                a.append(t / (2 * m))
        for community in partition:
            t = 0.0
            for i in range(len(community)):
                for j in range(len(community)):
                    if (G.has_edge(community[i], community[j])):
                        t += 1.0
            e.append(t / (2 * m))
        q = 0.0
        for ei, ai in zip(e, a):
            q += (ei - ai ** 2)
        return q

    def draw_graph(edge_index, x=None, y=None, title=None):
        # 创建一个无向图对象
        G = nx.Graph()

        # 添加边到图中
        for edge in edge_index.t().tolist():
            G.add_edge(edge[0], edge[1])

       # G.nodes()得到的点不一定是顺序的。
        node_order = torch.tensor(list(G.nodes()))

        x = x[node_order]

        # 使用不同的布局算法布置节点位置
        pos = nx.kamada_kawai_layout(G)
        # pos = nx.spring_layout(G)  # 弹簧布局，模拟了弹簧和质点系统的物理特性
        # pos = nx.spectral_layout(G)  # 谱布局，尤其在一些大型复杂图形上可能表现良好。不太好，点少也会聚在一起
        if x is not None:
            # 获取节点的特征值
            node_colors = [item.item() for item in x]  # Convert tensor to list

            # 绘制节点，离散的颜色表示
            cmap = plt.get_cmap('Paired', max(node_colors) + 1)

            nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=100, node_color=node_colors, cmap=cmap,
                    font_color='black', font_size=6, vmin=min(node_colors), vmax=max(node_colors))

            # 显示图的类别信息
            plt.text(0.95, 0.05, f'Graph Class: {y.item()}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='center', horizontalalignment='right')

            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm.set_array([])
            plt.colorbar(sm, ticks=range(min(node_colors), max(node_colors) + 1), label='Node Feature Index', shrink=0.9)

            # 显示图形
            plt.suptitle(title, fontsize=12)
            plt.show()
        else:
            # 绘制图形
            nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=100, node_color='skyblue',
                    font_color='black', font_size=8)

            # 显示图的类别信息
            plt.text(0.95, 0.05, f'Graph Class: {y.item()}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='center', horizontalalignment='right')

            # 显示图形
            plt.suptitle(title, fontsize=12)
            plt.show()

    def extract(self, data):
        """
        data: PyG的Data对象
        返回: List[List[int]], 每个元素是子图中节点的原始索引
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        remaining_nodes = torch.arange(num_nodes)  # 记录还没被提取的节点
        subgraphs = []

        for _ in range(6):
            if remaining_nodes.numel() == 0:
                break  # 没节点可提取了

            # 计算当前剩余节点的度
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[remaining_nodes] = True
            valid_edges = mask[edge_index[0]] & mask[edge_index[1]]
            filtered_edge_index = edge_index[:, valid_edges]
            degrees = torch.bincount(filtered_edge_index.flatten(),
                                     minlength=num_nodes)

            # 在剩余节点里选最大度节点
            degrees_remaining = degrees[remaining_nodes]
            max_idx = torch.argmax(degrees_remaining)
            center_node = remaining_nodes[max_idx].unsqueeze(0)

            # k跳子图
            sub_nodes, _, _, _ = k_hop_subgraph(
                node_idx=center_node,
                num_hops=2,
                edge_index=edge_index,
                relabel_nodes=False
            )

            subgraphs.append(sub_nodes.tolist())  # 转为list

            # 更新剩余节点：移除当前子图节点
            remaining_nodes = remaining_nodes[~torch.isin(remaining_nodes, sub_nodes)]

        return subgraphs