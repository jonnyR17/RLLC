import numpy as np
import time
import itertools
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from torch_geometric.utils import *
import networkx as nx
from torch_geometric.data import Batch
import tqdm
from QNetwork import Memory, EstimatorNetwork, Selector
from collections import namedtuple
from copy import deepcopy
import torch_geometric as pyg
import random
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import f1_score
from collections import deque
import matplotlib as plt
import community as community_louvain  # pip install python-louvain
from cdlib import algorithms

class NodeSelector(Selector):
    def __init__(self, *args, **kwargs):
        super(NodeSelector, self).__init__(*args, **kwargs)
        self.qnet = EstimatorNetwork(2,
                                     self.state_shape,
                                     self.mlp_layers,
                                     self.device)
        self.qnet.eval()
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self.memory = deque(maxlen=self.replay_memory_size)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0")



    def predict(self, node_list, graph_node_embedding, graph, batch):
        """

        Args:
            node_list: node_list
            graph_node_embedding:graph_node_embedding
            graph: graph
            batch: index

        Returns:
            node_prob:子图中心节点的选择概率[p(0)，p(1)]
        """
        """
               首先生成仅由candidate_node_list1中节点及其邻居构成的图，
               candidate_sub_node,candidate_edge = k_hop_subgraph(candidate_node_list1,3,graph.edge_index)
               fake_data = pyg.data.Data(x=candidate_sub_node,edge_index=candidate_edge)
               添加LPA方法进行社区发现，以candidate_node_list为聚类中心，添加标签。
               node_fake_lable = LPA(fake_data)
               fake_data.append(pyg.data.Data(x=node_fake_lable)
               后通过模块度算法进行聚类评价。
               MQ=Module(fake_data,candidate_node_list1)
               通过fed_reward_node 更新奖励。
               fed_reward_node(MQ)

        """
        CandidateNode_embeding_list = []
        flag=[]
        rw = False
        for candidate in node_list:
            """K_hop_subgraph(节点编号，k跳数，节点边)输出为（k跳邻居节点编号列表（包括自身节点），子图连边编号，映射值，连边遮掩）"""
            one_hop_nodes = set(k_hop_subgraph(int(candidate),3 ,graph.edge_index)[0].numpy()) - set([int(candidate)])#去除自身节点的3阶邻居节点集合
            CandidateNode_embeding_list.append(graph_node_embedding[list(one_hop_nodes)].mean(0))#将3阶邻居的嵌入按顺序添加保存。
        Candidate_node_embeding = torch.cat((graph_node_embedding[node_list], torch.stack(CandidateNode_embeding_list)),dim=1)#将候选节点自身嵌入与3阶邻居嵌入拼接起来。
        self.qnet.eval()
        with torch.no_grad():#设置反向传播时不会自动求导
            node_prob = self.qnet(Candidate_node_embeding)#将嵌入值输入q网络中得到预测值[p(0)，p(1)]
            node_prob = F.softmax(node_prob,dim=1)#进行softmax归一化概率
        assert len(torch.isnan(node_prob).nonzero()) == 0
        candidate_node_list1 = np.random.choice(node_list,1,replace=False)#对所有的候选节点选择概率最高的动作概率

        for i in range(len(candidate_node_list1)):  # 将所有的选中节点提取出
            if candidate_node_list1[i] == 1:
                flag.append(i)
        candidate_node_list = list(set(flag))#去除重复

        if len(node_list) <=20:
            ns = len(node_list)
            candidate_node_list=np.random.choice(node_list, ns, replace=False)
            rw = True

        candidate_node_list = torch.tensor(candidate_node_list, dtype=torch.long)
        # community = extract_root_subgraphs(graph,candidate_node_list,k=2)
        community = nx.community.asyn_lpa_communities(graph)
        if rw:
            reward = -1
        else:
            reward = self.Cal_Q(community, graph)
        if self._train:
            self.add_memory(Candidate_node_embeding,candidate_node_list1,reward)
        return candidate_node_list, community





    def add_memory(self, embedding, results, reward):
        self.memory.append((embedding, results, reward))

    def train(self, t):
        self.qnet.train()
        self.optimizer.zero_grad()

        state, action, reward = self.memory_sample(self.memory)
        state=state[0]
        action = action[0]
        action = torch.tensor(action).unsqueeze(1).to(self.device)
        t.eval()
        
        with torch.no_grad():
            target_action = t.forward(state)

        a = torch.argmax(target_action, dim=-1)
        r = reward[0]
        r = [r for i in range(len(a)) ]
        r = torch.tensor(r).to(self.device)
        target_action = target_action.to(self.device)
        if r.device != target_action.device:
            raise RuntimeError(f"different devices:r:{r.device},T:{target_action.device}")
        y = r + self.discount_factor * target_action.max(1)[0]
        y=y.to(self.device)
        q = self.qnet.forward(state).to(self.device)

        Q = q.gather(1,action).to(self.device)
        y = y.unsqueeze(1)
        # print(Q,y)
        loss = self.mse_loss(Q, y).to(self.device)
        self.optimizer.step()
        # print(f"agent_loss: {loss}")
        return loss.item()

    def memory_sample(self, memory):
        if len(memory) < self.batch_size:
            return None, None, None
        T = random.sample(self.memory,1)
        batch = list(zip(*T))
        
        state_batch = batch[0]
        action_batch = batch[1]
        reward_batch = batch[2]
        return state_batch, action_batch, reward_batch
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
                if m == 0 :
                    m =1
                a.append(t / (2 * m))
        for community in partition:
            t = 0.0
            for i in range(len(community)):
                for j in range(len(community)):
                    if (G.has_edge(community[i], community[j])):
                        t += 1.0
            if m == 0:
                m = 1
            e.append(t / (2 * m))
        q = 0.0
        for ei, ai in zip(e, a):
            q += (ei - ai ** 2)
        return q
    
    def is_full(self):
        return len(self.memory) == self.replay_memory_size

def extract_root_subgraphs(graph, candidate_node_list, k=3, num_subgraphs_per_root=1):
    """
    Args:
        graph: PyG 图对象
        candidate_node_list: 根节点列表
        k: k-hop 邻居
        num_subgraphs_per_root: 每个根节点生成的子图数
    Returns:
        communities: List[List[int]]，与 nx.community.asyn_lpa_communities 输出格式相同
    """
    communities = []

    for root_node in candidate_node_list:
        for _ in range(num_subgraphs_per_root):
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(root_node, k, graph.edge_index)
            # subset 就是节点集合
            communities.append(subset.tolist())

    return communities