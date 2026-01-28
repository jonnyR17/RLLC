import time
import argparse
import json

from tqdm import tqdm
import numpy as np
import os

from datetime import datetime
from model.data import load_dataset, kfolds, graph_kfolds
from model.Sugar import Net
from model.train_eval import train, test

from model.node_selector_lite import NodeSelector
from model.agent_chain import AgentChain
from toolbox.MetricSave import FoldMetricBase
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.transforms import BaseTransform

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
import torch_geometric.transforms as T

from sklearn.model_selection import StratifiedKFold


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def init_args(user_args=None):
    sub_num_dict = {
        "MUTAG": 5,
        "PTC_MR": 14,
        "PROTEINS": 40,
        "DD": 285,
        "NCI1": 30,
        "NCI109": 30,
        "ENZYMES": 33,
        "IMDB-BINARY": 20,
        "COLLAB": 75,
        "IMDB-MULTI": 20,
        "REDDIT-BINARY": 20,
        "ZINC": 2
    }
    parser = argparse.ArgumentParser(description='RLLC')
    parser.add_argument('--node_selector_action', type=int, default=50)
    parser.add_argument('--ablation_depth', type=int, default=0)
    parser.add_argument('--subgraph_num_delta', type=int, default=0)

    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--comment', type=str, default='debug')
    parser.add_argument('--tb', type=int, default=0, help="enable the tensorboard")

    # parser.add_argument('--dataset', type=str, default="MUTAG")
    # parser.add_argument('--dataset', type=str, default="PTC_MR")
    # parser.add_argument('--dataset', type=str, default="COLLAB")
    parser.add_argument('--dataset', type=str, default="REDDIT-BINARY")
    # parser.add_argument('--dataset', type=str, default="PROTEINS")
    # parser.add_argument('--dataset', type=str, default="NCI109")
    # parser.add_argument('--dataset', type=str, default="NCI1")
    # parser.add_argument('--dataset', type=str, default="DD")
    # parser.add_argument('--dataset', type=str, default="IMDB-MULTI")
    # parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    # parser.add_argument('--dataset', type=str, default="REDDIT-MULTI-5K")

    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--RL', type=int, default=0)
    parser.add_argument('--save_RL', type=int, default=1)
    parser.add_argument('--task', type=str, default='test')

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--GNN_episodes', type=int, default=100)
    parser.add_argument('--RL_episodes', type=int, default=50)
    parser.add_argument('--agent_episodes', type=int, default=150)

    parser.add_argument('--max_timesteps', type=int, default=5)
    parser.add_argument('--RL_lr', type=float, default=0.001)
    parser.add_argument('--RL_weight_decay', type=float, default=0.0001)
    parser.add_argument('--RL_batch_size', type=int, default=64)

    # coeff
    parser.add_argument('--sub_coeff', type=float, default=0.2)
    parser.add_argument('--mi_coeff', type=float, default=0.5)

    # RL
    parser.add_argument('--pretrain_agent', type=int, default=1)  # 1=先预训练智能体
    parser.add_argument('--pretrain_agent_epochs', type=int, default=100)  # 预训练跑几轮 loader
    parser.add_argument('--finetune_agent', type=int, default=0)  # 训练 Sugar 时是否继续训练智能体
    parser.add_argument('--agent_ckpt_dir', type=str, default='./best_save/agent_ckpt')
    # ===== Reward experiment =====
    parser.add_argument('--reward_exp', type=int, default=0, help='1=log reward to csv during agent calls')
    parser.add_argument('--reward_only', type=int, default=0, help='1=only run global agent pretrain then exit')
    parser.add_argument('--reward_log_dir', type=str, default='./reward_logs')
    parser.add_argument('--reward_log_flush', type=int, default=200)

    parser.add_argument('--replay_memory_size', type=int, default=100)
    parser.add_argument('--update_target_estimator_every', type=int, default=5)
    parser.add_argument('--mlp_layers', type=list, default=[64, 128, 256, 128, 64])
    parser.add_argument('--action_num', type=int, default=3)
    parser.add_argument('--fixed_k_hop', type=int, default=1)
    parser.add_argument('--discount_factor', type=float, default=0.95)
    parser.add_argument('--epsilon_start', type=float, default=1.)
    parser.add_argument('--epsilon_end', type=float, default=0.2)
    parser.add_argument('--epsilon_decay_steps', type=int, default=100)
    parser.add_argument('--norm_step', type=int, default=200)
    parser.add_argument('--hid_dim', type=int, default=128)

    args = parser.parse_args()
    if user_args is not None:
        for k, v in user_args.items():
            setattr(args, k, v)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not hasattr(args, 'sub_num'):
        setattr(args, 'sub_num', sub_num_dict[args.dataset] + args.subgraph_num_delta)

    return args

class AddPE(BaseTransform):
    def __init__(self, walk_length=20):
        self.walk_length = walk_length

    def __call__(self, data):
        N = data.num_nodes
        # 假设生成随机向量作为 PE
        data.pe = torch.randn(N, self.walk_length)
        return data

def init_metric_saver(folds, dir_name, file_name, debug=False):
    metric_saver = FoldMetricBase(k_fold=folds, dir_name=dir_name, file_name=file_name, tb_server=debug)
    return metric_saver
def freeze_agent(agent):
    """让agent只推理不训练：eval模式 + 禁止梯度"""
    agent._eval()
    qnet = agent.node_selector.qnet
    qnet.eval()
    for p in qnet.parameters():
        p.requires_grad_(False)


def pretrain_agent_on_loader(train_loader, model, agent, args, ckpt_path):
    """
    用训练集跑若干遍，让agent把经验装进replay memory并更新Q网络。
    注意：这里不更新GNN参数。
    """
    device = args.device
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # 冻结GNN，避免误更新/构图开销
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    agent._train()

    for ep in range(args.pretrain_agent_epochs):
        for graph in train_loader:
            graph = graph.to(device)

            # 只需要触发 agent.predict(...) 来收集记忆，不需要GNN梯度
            with torch.no_grad():
                _ = model(graph, agent)

            # 满了就更新一次Q网络
            if agent.is_full():
                agent.train()
                agent.clear()

    # 训练结束：保存qnet参数
    torch.save(agent.node_selector.qnet.state_dict(), ckpt_path)

    # 恢复GNN可训练
    for p in model.parameters():
        p.requires_grad_(True)

    agent._eval()
    return ckpt_path

def pretrain_agent_once(train_dataset, graphs, args, ckpt_path):
    """
    用 train_dataset（global union train）预训练一次 agent，并保存 qnet。
    显示 tqdm 进度条（epoch + batch）。
    """
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    model_tmp, agent_tmp, _ = init_model(graphs, args)

    batch_size = args.batch_size
    if 'adj' in train_dataset[0]:
        train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # 冻结GNN
    model_tmp.eval()
    for p in model_tmp.parameters():
        p.requires_grad_(False)

    agent_tmp._train()

    tqdm.write(
        f"[GLOBAL PRETRAIN] >>> Start pretraining agent "
        f"(train_size={len(train_dataset)}, epochs={args.pretrain_agent_epochs}, batch={batch_size})"
    )

    t0 = time.perf_counter()
    update_steps = 0

    # 外层 epoch 条
    ep_bar = tqdm(range(args.pretrain_agent_epochs), desc="PretrainAgent Epoch", leave=True)
    for ep in ep_bar:
        for batch in train_loader:
            batch = batch.to(args.device)

            with torch.no_grad():
                _ = model_tmp(batch, agent_tmp)

            if agent_tmp.is_full():
                agent_tmp.train()
                agent_tmp.clear()

            # 动态显示一些信息（可删）
            # mem_len = len(agent_tmp.node_selector.memory) if hasattr(agent_tmp, "node_selector") else 0
            # batch_bar.set_postfix({
            #     "updates": update_steps,
            #     "mem": mem_len
            # })

        # ep_bar.set_postfix({"updates": update_steps})

    # 保存 qnet
    torch.save(agent_tmp.node_selector.qnet.state_dict(), ckpt_path)
    cost = time.perf_counter() - t0

    tqdm.write(f"[GLOBAL PRETRAIN] >>> Finished in {cost:.2f}s. Saved: {ckpt_path}")

    del model_tmp, agent_tmp
    torch.cuda.empty_cache()

    return ckpt_path
def load_dataset(dataset_name="MUTAG"):
    transform = None  # ✅ 关键：先定义，避免 REDDIT-BINARY 分支未赋值

    if dataset_name != "REDDIT-BINARY":
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        graphs = TUDataset(root="./data/raw", name=dataset_name, transform=transform)
    else:
        graphs = TUDataset(root="./data/raw", name=dataset_name)

    if graphs.data.x is None:
        max_degree = 0
        mean = 0
        std = 0
        for data in graphs:
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(deg.max()))
            mean += deg.float().mean()
            std += deg.float().std()
        mean = mean / len(graphs)
        std = std / len(graphs)

        if dataset_name == "PROTEINS":
            x_transform = NormalizedDegree(mean, std)
        else:
            x_transform = T.OneHotDegree(max_degree)

        graphs.transform = T.Compose([transform, x_transform]) if transform is not None else x_transform

    for i, g in enumerate(graphs):
        g.graph_id = int(i)
    return graphs



def init_model(graphs, args, k_fold=10):
    net = Net(max_layer=3,
              node_dim=graphs[0].x.shape[1],
              hid_dim=args.hid_dim,
              out_dim=graphs.num_classes,
              sub_num=args.sub_num,
              sub_size=15,
              loss_type=0,
              sub_coeff=args.sub_coeff,
              mi_coeff=args.mi_coeff,

              device=args.device)
    net.to(args.device)

    node_selector = NodeSelector(
        action_num=args.node_selector_action,
        fixed_k_hop=args.fixed_k_hop,
        lr=args.RL_lr,
        batch_size=args.RL_batch_size,
        state_shape=args.hid_dim,
        mlp_layers=args.mlp_layers,
        replay_memory_size=args.replay_memory_size,
        discount_factor=args.discount_factor,
        device=args.device,

        # ===== Reward experiment =====
        reward_exp=getattr(args, "reward_exp", 0),
        reward_only=getattr(args, "reward_only", 0),
        reward_log_dir=getattr(args, "reward_log_dir", "./reward_logs"),
        reward_log_flush=getattr(args, "reward_log_flush", 100),
        dataset=getattr(args, "dataset", "DATA"),
    )

    agent = AgentChain(update_target_estimator_every=args.update_target_estimator_every,
                       time_step=18,
                       max_k_hop=args.action_num,
                       epochs=args.agent_episodes,
                       ablation_depth=args.ablation_depth).bind_selector(candidate=node_selector)
    # agent.visual = 1
    # agent.bind_selector(depth=depth_selector, neighbor=neighbor_selector)

    optimizer = torch.optim.Adam(net.parameters(),
                                 args.lr,
                                 weight_decay=args.weight_decay)
    return net, agent, optimizer
def build_global_train_idx(graphs, folds):
    """
    返回：所有fold中 train_idx 的并集（sorted list[int]）
    """
    # k_fold(graphs, folds) 需要返回 (train_splits, test_splits, val_splits)
    train_splits, test_splits, val_splits = k_fold(graphs, folds)

    union_set = set()
    for tr in train_splits:
        # tr 可能是 numpy array / torch tensor / list
        if hasattr(tr, "tolist"):
            tr = tr.tolist()
        union_set.update(list(tr))

    global_train_idx = sorted(union_set)
    return global_train_idx


def train_GNN(args, folds, graphs, metric_saver, time_str):
    best = []

    # ========== ① 先构造“全局训练集（所有fold train_idx并集）” ==========
    global_train_idx = build_global_train_idx(graphs, folds)
    global_train_dataset = graphs[global_train_idx]
    print(f"[GLOBAL TRAIN] size = {len(global_train_dataset)} (union of all fold train sets)")

    # ========== ② 只预训练一次 agent，并保存 ==========
    ckpt_dir = getattr(args, "agent_ckpt_dir", "./agent_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    # global_ckpt = os.path.join(ckpt_dir, f"IMDB-BINARY_AGENT_GLOBALTRAIN.pt")
    global_ckpt = os.path.join(ckpt_dir, f"{args.dataset}_AGENT_GLOBALTRAIN.pt")

    # ========== ② 只预训练一次 agent，并保存（若已存在则跳过） ==========
    if getattr(args, "pretrain_agent", 1):
        if os.path.exists(global_ckpt):
            tqdm.write(f"[GLOBAL PRETRAIN] >>> Found existing agent ckpt, SKIP pretrain: {global_ckpt}")
        else:
            tqdm.write(f"[GLOBAL PRETRAIN] >>> No ckpt found, start pretrain and save to: {global_ckpt}")
            pretrain_agent_once(global_train_dataset, graphs, args, global_ckpt)
            # ===== Reward experiment: only pretrain agent and exit =====
            if getattr(args, "reward_only", 0):
                print(
                    f"[REWARD ONLY] Finished global pretrain. Logs in: {getattr(args, 'reward_log_dir', './reward_logs')}")
                return

    else:
        tqdm.write("[GLOBAL PRETRAIN] >>> pretrain_agent=0, skip pretrain by config.")

    # ========== ③ 每个fold：训练 GNN，但 agent 统一load同一个 ckpt 并冻结 ==========
    # 注意：fold 内 train/test 仍按 k_fold 划分（你的要求：训练GNN还是按fold做）
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(graphs, folds))):
        print(f"============================{fold+1}/{folds}==================================")

        train_dataset = graphs[train_idx]
        test_dataset  = graphs[test_idx]
        batch_size = args.batch_size

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            test_loader  = DenseLoader(test_dataset,  batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            test_loader  = DataLoader(test_dataset,  batch_size, shuffle=False)

        model, agent, optimizer = init_model(graphs, args)

        # load 同一个全局 agent 权重（存在就加载，不存在就用随机初始化并冻结）
        if os.path.exists(global_ckpt):
            sd = torch.load(global_ckpt, map_location=args.device)
            agent.node_selector.qnet.load_state_dict(sd)
            agent.node_selector.qnet.to(args.device)
            tqdm.write(f"[Fold {fold + 1}/{folds}] >>> Loaded global agent ckpt: {global_ckpt}")
        else:
            tqdm.write(f"[Fold {fold + 1}/{folds}] >>> WARNING: no agent ckpt found, agent stays random.")

        freeze_agent(agent)  # 关键：保证训练阶段不更新 agent

        trange = tqdm(range(1, args.GNN_episodes + 1))
        best_test = 0.0

        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-6,
        )

        n_train = len(train_dataset)
        n_test  = len(test_dataset)

        for i_episode in trange:
            # ---- train timing ----
            t0 = time.perf_counter()
            train_acc, train_loss = train(train_loader, model, agent, optimizer, device=args.device)
            t1 = time.perf_counter()

            # ---- test timing ----
            t2 = time.perf_counter()
            test_acc, test_loss = test(test_loader, model, agent, device=args.device)
            t3 = time.perf_counter()

            train_elapsed = t1 - t0
            test_elapsed  = t3 - t2

            if test_acc > best_test:
                best_test = test_acc

            scheduler.step(test_acc)

            train_ms_g = 1000.0 * train_elapsed / max(1, n_train)
            test_ms_g  = 1000.0 * test_elapsed  / max(1, n_test)

            trange.set_postfix({
                "train_acc": f"{train_acc:.4f}",
                "train_loss": f"{train_loss:.4f}",
                "test_acc": f"{test_acc:.4f}",
                "test_loss": f"{test_loss:.4f}",
                "best_acc": f"{best_test:.4f}",
                "train_ms/g": f"{train_ms_g:.2f}",
                "test_ms/g": f"{test_ms_g:.2f}",
            })

            metric_saver.add_record(train_acc, train_loss, test_acc, test_loss)

        best.append(best_test)

    mean = np.mean(best)
    std  = np.std(best)
    print("best", mean, "\\pm", std)

def normalize_list(lst):
    """
    将列表归一化到 [0, 1]，最小值映射为0，最大值映射为1
    """
    min_val = min(lst)
    max_val = max(lst)
    if max_val == min_val:
        # 避免除以0，当所有元素相等时返回0列表
        return [0 for _ in lst]
    return [(x - min_val) / (max_val - min_val) for x in lst]

def save(fold, args, metric_saver, best_node_selector_net, model, time):
    time_str = time
    args.device = 0
    save_dir = os.path.join('./best_save/RLLC/', args.dataset, "fold_{}".format(fold),
                            "{:.5f}".format(metric_saver.cur_saver.strict_best_acc))
    metric_saver.cur_saver.save(save_dir, prefix='results: ')
    with open(save_dir + "/args.json", 'w') as f:
        json.dump(args.__dict__, f)

    # best_node_selector_net.save(save_dir+'/node.pt')
    # model.save(save_dir)
    print(f"save to {save_dir}")


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), [int(x) for x in dataset.data.y]):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        # train_indices.append(balance(dataset.data.y, train_mask.nonzero(as_tuple=False).view(-1)))
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    return train_indices, test_indices, val_indices


def meta_info(graphs):
    iso, size = [], []
    for index, graph in enumerate(graphs):
        if contains_isolated_nodes(graph.edge_index):
            iso.append(index)
        size.append(graph.x.shape[0])
    print(graphs)
    iso_np, size_np = np.array(iso), np.array(size)
    print(
        f"{graphs.name}: max_size:{size_np.max()} min_size:{size_np.min()} avg_size:{size_np.mean()} node_label:{graph.x.shape[-1]}")


def main(args=None, k_fold=10):
    if args is None:
        args = init_args()
    graphs = load_dataset(dataset_name=args.dataset)
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    meta_info(graphs)
    metric_saver = init_metric_saver(args.folds, 'test', args.comment, debug=args.tb)
    train_GNN(args, k_fold, graphs, metric_saver, time_str)


import matplotlib.pyplot as plt


def plot_list(data, plot_type='line', title='List Data Visualization', xlabel='Index', ylabel='Value'):

    plt.figure(figsize=(10, 5))

    if plot_type == 'line':
        plt.plot(data, marker='o', linestyle='-', color='b', label='Value')
        plt.legend()
    elif plot_type == 'bar':
        plt.bar(range(len(data)), data, color='skyblue')
    else:
        raise ValueError("plot_type must be 'line' or 'bar'")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 设置横坐标每10个显示一个
    xticks = range(0, len(data), 10)
    plt.xticks(xticks)

    # 去掉背景网格
    plt.grid(False)
    plt.show()




if __name__ == "__main__":
    main()
