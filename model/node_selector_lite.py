import sys

from community import community_louvain
import os
import csv
import time

import torch.nn as nn

from .QNetwork import Memory, EstimatorNetwork, Selector

import random
import torch.nn.functional as F



class NodeSelector(Selector):
    def __init__(self, *args, **kwargs):
        super(NodeSelector, self).__init__(*args, **kwargs)
        self.qnet = EstimatorNetwork(2,
                                     self.state_shape,
                                     self.mlp_layers,
                                     self.device)

        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
        self.memory = deque(maxlen=self.replay_memory_size)
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.qnet.parameters(), lr=self.lr)
        # ===== Reward experiment logger =====
        self._reward_step = 0
        self._best_by_graph = {}   # graph_id -> best_so_far (empirical ceiling)
        self._reward_rows_since_flush = 0
        self._reward_csv_fp = None
        self._reward_csv_writer = None

        if getattr(self, "reward_exp", 0):
            log_dir = getattr(self, "reward_log_dir", "./reward_logs")
            os.makedirs(log_dir, exist_ok=True)
            run_tag = time.strftime("%Y%m%d-%H%M%S")
            ds = getattr(self, "dataset", "DATA")
            log_path = os.path.join(log_dir, f"{ds}_{run_tag}.csv")

            self._reward_csv_fp = open(log_path, "w", newline="", encoding="utf-8")
            self._reward_csv_writer = csv.writer(self._reward_csv_fp)
            self._reward_csv_writer.writerow([
                "step", "graph_id", "batch_local",
                "n_nodes", "n_edges", "n_seeds", "n_comms",
                "reward", "best_so_far", "gap_best"
            ])
            self._reward_csv_fp.flush()
            print(f"[RewardExp] logging to: {log_path}")


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
        flag=[]
        rw = False
        Candidate_node_embeding = graph_node_embedding
        self.qnet.eval()
        with torch.no_grad():#设置反向传播时不会自动求导
            node_prob = self.qnet(Candidate_node_embeding)#将嵌入值输入q网络中得到预测值[p(0)，p(1)]
            node_prob = F.softmax(node_prob,dim=1)#进行softmax归一化概率
        num_nodes = graph.num_nodes  # 注意没有括号
        # k = 3  # 预算k（你可以改成参数）
        #
        # # 1) 先按动作预测筛出 action=1 的节点
        # candidate_node_list1 = torch.argmax(node_prob, dim=1)  # [N] -> 0/1
        # pos_idx = torch.where(candidate_node_list1 == 1)[0]  # action=1 的节点索引
        #
        # device = node_prob.device
        #
        # if pos_idx.numel() == 0:
        #     # 保持格式一致：空的 LongTensor
        #     candidate_node_list = torch.empty(0, dtype=torch.long, device=device)
        # else:
        #     score = node_prob[pos_idx, 1]  # action=1 概率
        #     k_eff = min(k, score.numel())
        #     topk_pos = torch.topk(score, k_eff, largest=True).indices
        #     candidate_node_list = pos_idx[topk_pos].long()  # 映射回全局节点索引list

        candidate_node_list1 = torch.argmax(node_prob,dim=1)#对所有的候选节点选择概率最高的动作概率
        candidate_node_list = torch.where(candidate_node_list1 == 1)[0]
        # community = community_detection_with_seeds(graph.edge_index,graph.num_nodes,candidate_node_list.tolist())
        community = community_detection_with_seeds(graph.edge_index,graph.num_nodes,candidate_node_list)



        community = list(community)  # 将得到的节点转化为列表

        if len(community)==0:
            print(community)
        G = to_networkx(graph)
        # candidate_node_list: list[int]
        # max_node_idx = max(candidate_node_list)
        # num_nodes_in_G = G.number_of_nodes()  # NetworkX Graph 获取节点数
        #
        # if max_node_idx >= num_nodes_in_G:
        #     raise ValueError(
        #         f"Candidate node index {max_node_idx} exceeds graph node count {num_nodes_in_G}. "
        #         f"Candidate nodes: {candidate_node_list}"
        #     )
        # reward = modularity(G,community)
        if candidate_node_list.numel() != 0:
            reward =  average_node_modularity(G, candidate_node_list, community)*10

            # c_hat = compute_c_hat_from_nx(G, kind="pagerank")  # 或 "degree"/"coreness"
            # union_nodes = sorted({u for comm in community for u in comm}) if community else []
            # reward = centrality_reward(c_hat, union_nodes, lam=0.3) * 10

        else:
            reward = -1
        # reward =  modularity_reward(G, candidate_node_list)*10
        if self._train:
            self.add_memory(Candidate_node_embeding,candidate_node_list1,reward)
        # ===== Reward experiment: log reward + empirical ceiling (best-so-far) =====
        if getattr(self, "reward_exp", 0) and (self._reward_csv_writer is not None):
            self._reward_step += 1

            gid = int(getattr(graph, "graph_id", batch))
            n_nodes = int(graph.num_nodes)
            n_edges = int(graph.edge_index.size(1)) if hasattr(graph, "edge_index") else 0
            n_seeds = int(candidate_node_list.numel()) if hasattr(candidate_node_list, "numel") else int(len(candidate_node_list))
            n_comms = int(len(community)) if community is not None else 0

            r = float(reward)
            best = self._best_by_graph.get(gid, -1e18)
            if r > best:
                best = r
                self._best_by_graph[gid] = best
            gap = best - r

            self._reward_csv_writer.writerow([
                self._reward_step, gid, int(batch),
                n_nodes, n_edges, n_seeds, n_comms,
                r, best, gap
            ])
            self._reward_rows_since_flush += 1
            if self._reward_rows_since_flush >= int(getattr(self, "reward_log_flush", 200)):
                self._reward_csv_fp.flush()
                self._reward_rows_since_flush = 0

        return candidate_node_list, community

    def add_memory(self, embedding, results, reward):
        """Store transitions on CPU to avoid GPU memory blow-up."""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu()
        if isinstance(results, torch.Tensor):
            results = results.detach().cpu()
        # reward 通常是标量
        try:
            reward = float(reward)
        except Exception:
            pass
        self.memory.append((embedding, results, reward))

    def memory_sample(self, memory):
        """Sample a mini-batch from replay memory.

        Note: TU 图的节点数不一致，这里先返回一个 list of transitions，
        train() 里逐条计算 loss 再取平均。
        """
        if len(memory) == 0:
            return None, None, None
        n = min(self.batch_size, len(memory))
        T = random.sample(memory, n)
        batch = list(zip(*T))
        state_batch = batch[0]  # list[tensor], each [num_nodes, state_dim]
        action_batch = batch[1]  # list[tensor/list], each [num_nodes] with 0/1 actions
        reward_batch = batch[2]  # list[float] (scalar reward per graph)
        return state_batch, action_batch, reward_batch

    def train(self, target_net):
        """One DQN update using replay memory."""
        state_batch, action_batch, reward_batch = self.memory_sample(self.memory)
        if state_batch is None:
            return 0.0

        self.qnet.train()
        target_net.eval()
        self.optimizer.zero_grad()

        losses = []
        for state, action, reward in zip(state_batch, action_batch, reward_batch):
            state = state.to(self.device)  # [num_nodes, state_dim]

            # action: [num_nodes] -> [num_nodes,1]
            if isinstance(action, torch.Tensor):
                action_t = action.long().view(-1, 1)
            else:
                action_t = torch.tensor(action, dtype=torch.long).view(-1, 1)
            action_t = action_t.to(self.device)

            # reward: scalar -> broadcast to [num_nodes,1]
            reward_t = torch.tensor(float(reward), dtype=torch.float, device=self.device).view(1, 1)

            with torch.no_grad():
                target_q = target_net(state)  # [num_nodes, num_actions]
                max_target_q, _ = target_q.max(dim=1, keepdim=True)  # [num_nodes,1]
                y = reward_t + self.discount_factor * max_target_q  # broadcast

            q = self.qnet(state)  # [num_nodes, num_actions]
            q_selected = q.gather(1, action_t)  # 用 replay 里的 action
            loss = F.mse_loss(q_selected, y)
            losses.append(loss)

        loss_q = torch.stack(losses).mean()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss_q.item())

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

def community_detection_with_seeds(edge_index, num_nodes, seed_nodes):
    """
    使用种子节点进行社区划分，返回社区列表，每个社区是节点索引列表
    """
    # 转换为 NetworkX 图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # 初始分区：种子节点强制在社区0
    partition_init = {}
    for node in range(num_nodes):
        if node in seed_nodes:
            partition_init[node] = 0
        else:
            partition_init[node] = -1  # 未分配

    # Louvain 社区划分
    num_nodes = G.number_of_nodes()

    if num_nodes <= 17:
        partition = community_louvain.best_partition(G, partition=partition_init, random_state=42,resolution=1)
    else:
        partition = community_louvain.best_partition(G, partition=partition_init, random_state=42,resolution=1)
    # 将 partition 转成社区列表
    community_dict = {}
    for node, comm_id in partition.items():
        community_dict.setdefault(comm_id, []).append(node)

    # 转成列表
    communities = list(community_dict.values())
    return communities

import torch
from torch_geometric.utils import to_networkx
import networkx as nx
from collections import deque
def modularity_reward(G, selected):
    s = set(selected)
    other = [n for n in G.nodes if n not in s]
    if len(other) == 0: return 0.0
    return nx.algorithms.community.quality.modularity(G, [list(s), other])

def community_detection_seed_expansion(edge_index, num_nodes, seed_nodes):
    """
    基于种子节点的社区扩展划分
    Args:
        edge_index: torch.Tensor (2, num_edges)
        num_nodes: int, 节点总数
        seed_nodes: list[int], 种子节点
    Returns:
        communities: list of lists，每个子列表是社区节点
    """
    # 转 NetworkX 图

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # 初始化
    node_assigned = [-1] * num_nodes  # -1 表示未分配
    communities = [[] for _ in seed_nodes]

    # 将种子节点加入对应社区
    for idx, seed in enumerate(seed_nodes):
        node_assigned[seed] = idx
        communities[idx].append(seed)

    # 用队列做 BFS 扩展
    queue = deque(seed_nodes)
    while queue:
        node = queue.popleft()
        comm_id = node_assigned[node]
        for neighbor in G.neighbors(node):
            if node_assigned[neighbor] == -1:
                # 将邻居分配给当前社区
                node_assigned[neighbor] = comm_id
                communities[comm_id].append(neighbor)
                queue.append(neighbor)


    return communities


import networkx as nx


def average_node_modularity(G, nodes, communities):
    """
    计算指定节点列表的模块度平均值

    参数:
    G : nx.Graph
        网络图
    nodes : list
        待计算的节点列表
    communities : list of sets
        每个社区是一个节点集合

    返回:
    avg_modularity : float
        节点列表的平均模块度
    """
    m = G.size(weight='weight')  # 总边权
    degrees = dict(G.degree(weight='weight'))
    node_scores = []

    # 计算每个节点的模块度贡献
    community_map = {}  # 节点 -> 所属社区
    for comm in communities:
        for n in comm:
            community_map[n] = comm

    for node in nodes:
        node_id = int(node)  # 将 tensor 转为 int
        comm = community_map[node_id]
        score = sum(
            (1 if G.has_edge(node_id, j) else 0) - (degrees[node_id] * degrees[j]) / (2 * m)
            for j in comm
        )
        score /= (2 * m)
        node_scores.append(score)

    # 返回平均值
    return sum(node_scores) / len(node_scores) if node_scores else 0


def expand_assign_multisource_bfs_scatter(edge_index, num_nodes, seed_nodes, max_hops=2, add_singletons=False):
    """
    精确的多源K-hop BFS 归属（无权图 hop 距离）
    - 归属规则：先到先得；同一轮冲突时取 seed_id 更小（稳定）
    """
    device = edge_index.device
    row, col = edge_index[0].long(), edge_index[1].long()

    # seeds 去重保序
    seeds = []
    seen = set()
    for s in seed_nodes:
        s = int(s)
        if 0 <= s < num_nodes and s not in seen:
            seeds.append(s); seen.add(s)
    if len(seeds) == 0:
        return ([[i] for i in range(num_nodes)] if add_singletons else []), \
               torch.full((num_nodes,), -1, device=device, dtype=torch.long)

    seeds_t = torch.tensor(seeds, device=device, dtype=torch.long)
    K = seeds_t.numel()
    INF = torch.iinfo(torch.long).max
    base = K + 1  # key = hop*base + cid

    assigned = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
    hop = torch.full((num_nodes,), -1, device=device, dtype=torch.long)

    # 初始化
    assigned[seeds_t] = torch.arange(K, device=device, dtype=torch.long)
    hop[seeds_t] = 0
    frontier = torch.zeros(num_nodes, device=device, dtype=torch.bool)
    frontier[seeds_t] = True

    for h in range(max_hops):
        active = frontier[row]
        if not torch.any(active):
            break

        src = row[active]
        dst = col[active]
        cand_cid = assigned[src]  # 都是 >=0

        # 对每个 dst 取最小 cid（同一层冲突取更小cid）
        min_cid = torch.full((num_nodes,), INF, device=device, dtype=torch.long)
        min_cid.scatter_reduce_(0, dst, cand_cid, reduce="amin", include_self=True)

        new_nodes = (assigned == -1) & (min_cid != INF)
        if not torch.any(new_nodes):
            break

        assigned[new_nodes] = min_cid[new_nodes]
        hop[new_nodes] = h + 1
        frontier = new_nodes  # 下一层只从新加入的点扩

    # communities
    communities = [[] for _ in range(K)]
    assigned_cpu = assigned.detach().cpu().tolist()
    for v, cid in enumerate(assigned_cpu):
        if cid >= 0:
            communities[cid].append(v)
        elif add_singletons:
            communities.append([v])

    return communities

import torch

def community_detection_with_seeds_R(edge_index, num_nodes, seed_nodes,
                                  walk_length=6, num_walks=20, top_m=300,
                                  ensure_seed=True):
    """
    随机游走版“社区划分”（可直接替换原 Louvain 版本）
    输入/输出格式保持一致：
      Input:
        edge_index: torch.Tensor [2, E]
        num_nodes: int
        seed_nodes: list[int] / Tensor
      Output:
        communities: list[list[int]]  # 每个社区是节点索引列表

    逻辑：
      - 每个 seed 作为一个“社区中心”
      - 从该 seed 跑 num_walks 条长度 walk_length 的随机游走
      - 统计访问频率，取 top_m 个节点作为该 seed 的社区
      - 节点可能同时出现在多个社区（如果你希望“互斥归属”，我也能给你互斥版）
    """
    # ---- 依赖检查 ----
    try:
        from torch_cluster import random_walk
    except Exception as e:
        raise ImportError("需要安装 torch-cluster 才能使用随机游走社区划分。") from e

    device = edge_index.device
    edge_index = edge_index.long()
    row, col = edge_index[0], edge_index[1]

    # ---- seed 去重 + 越界过滤（保序）----
    seeds = []
    seen = set()
    for s in seed_nodes:
        s = int(s)
        if 0 <= s < num_nodes and s not in seen:
            seeds.append(s)
            seen.add(s)

    if len(seeds) == 0:
        return []

    communities = []

    # ---- 每个 seed 生成一个社区 ----
    for s in seeds:
        starts = torch.full((num_walks,), s, device=device, dtype=torch.long)
        walk = random_walk(row, col, starts, walk_length=walk_length)  # [num_walks, walk_length+1]
        nodes = walk.reshape(-1)
        nodes = nodes[nodes >= 0]  # 防御性过滤

        if nodes.numel() == 0:
            comm = [s] if ensure_seed else []
            communities.append(comm)
            continue

        # 访问频率
        counts = torch.bincount(nodes, minlength=num_nodes)

        # 确保 seed 一定入选
        if ensure_seed:
            counts[s] += counts.max().clamp(min=1)

        m = min(int(top_m), num_nodes)
        top_nodes = torch.topk(counts, k=m, largest=True).indices
        top_nodes = top_nodes.unique()

        communities.append(top_nodes.detach().cpu().tolist())

    return communities

import networkx as nx
import torch

def _rank_normalize(scores: torch.Tensor) -> torch.Tensor:
    """rank 归一化到 [0,1]，跨图更稳定"""
    n = scores.numel()
    if n <= 1:
        return torch.zeros_like(scores)
    order = torch.argsort(scores)  # ascending
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(n, dtype=torch.float, device=scores.device)
    return ranks / (n - 1.0)


def compute_c_hat_from_nx(G, kind="pagerank") -> torch.Tensor:
    """
    从 networkx Graph/DiGraph 计算中心性并做 rank-normalize，返回 c_hat: [num_nodes] in [0,1] (CPU tensor)
    注意：coreness 需要无向图，会自动转 undirected。
    """
    n = G.number_of_nodes()
    if n == 0:
        return torch.zeros(0, dtype=torch.float)

    if kind == "degree":
        d = dict(G.degree())
        c = torch.tensor([float(d.get(i, 0.0)) for i in range(n)], dtype=torch.float)

    elif kind == "pagerank":
        pr = nx.pagerank(G, alpha=0.85)
        c = torch.tensor([float(pr.get(i, 0.0)) for i in range(n)], dtype=torch.float)

    elif kind == "coreness":
        Gu = G.to_undirected() if hasattr(G, "to_undirected") else G
        core = nx.core_number(Gu)
        c = torch.tensor([float(core.get(i, 0.0)) for i in range(n)], dtype=torch.float)

    elif kind == "betweenness":
        bc = nx.betweenness_centrality(G, normalized=True)
        c = torch.tensor([float(bc.get(i, 0.0)) for i in range(n)], dtype=torch.float)

    else:
        raise ValueError(f"Unknown centrality kind: {kind}")

    return _rank_normalize(c)


def centrality_reward(c_hat: torch.Tensor, nodes, lam=0.3) -> float:
    """
    reward = mean(c_hat[selected]) - lam * |selected|/n
    nodes: list[int] 或 tensor[int]
    """
    n = int(c_hat.numel())
    if n == 0:
        return 0.0

    if isinstance(nodes, torch.Tensor):
        idx = nodes.detach().cpu().long()
    else:
        idx = torch.tensor(list(nodes), dtype=torch.long)

    # 过滤越界
    idx = idx[(idx >= 0) & (idx < n)]
    if idx.numel() == 0:
        return 0.0

    cent = float(c_hat[idx].mean().item())
    size_pen = float(idx.numel()) / float(n)
    return cent - lam * size_pen
