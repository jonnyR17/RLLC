import sys
import time

from torch_geometric.utils import *
import networkx as nx
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

def learn(loader, model, agent, optimizer, recorder=None):
    """

    Args:
        loader: train_loader=DenseLoader(train_dataset,batch_size)
        model: NET=model.Sugar
        agent: AgentChain
        optimizer: Adam
        recorder: None

    Returns:
        learn_acc,learn_loss
    """
    agent._eval()
    model.train()
    total_mae = 0.0
    total_samples = 0
    total_loss = 0.0

    for graph in loader:
        graph = graph.cuda("cuda:0")
        optimizer.zero_grad()
        # 模型前向
        predicts, loss = model(graph, agent)

        loss.backward()
        optimizer.step()

        # 累计 MAE
        mae = torch.abs(predicts.view(-1) - graph.y.view(-1))
        total_mae += mae.sum().item()
        total_samples += graph.y.size(0)

        # 累计总 loss
        total_loss += loss.item() * graph.y.size(0)  # 按样本数加权

    # 平均 MAE 和平均 loss
    MAE_loss = total_mae / total_samples
    avg_loss = total_loss / total_samples

    return MAE_loss, avg_loss

def train(loader, model, agent, optimizer):

    """
     Args:
         loader: train_loader=DenseLoader(train_dataset,batch_size)
         model: NET=model.Sugar
         agent: AgentChain
         optimizer: Adam

     Returns:
         train_acc,train_loss
     """
    correct, total_loss = 0, 0
    agent._eval()
    model.train()
    for graph in loader:
        graph = graph
        graph = graph.cuda("cuda:0")
        optimizer.zero_grad()
        predicts, loss = model(graph, agent)
        loss.backward()
        correct_vector = predicts
        correct += correct_vector
        optimizer.step()
    train_acc = correct / len(loader)
    train_loss = total_loss / len(loader)

    return train_acc, train_loss

def eval(loader, model, agent):
    total_mae = 0.0
    total_samples = 0
    total_loss = 0.0
    agent._eval()
    model.eval()
    with torch.no_grad():
        for graph in loader:
            graph = graph
            graph = graph.cuda("cuda:0")
            predicts, loss = model(graph, agent)
            mae = torch.abs(predicts.view(-1) - graph.y.view(-1))
            total_mae += mae.sum().item()
            total_samples += graph.y.size(0)

            # 累计总 loss
            total_loss += loss.item() * graph.y.size(0)  # 按样本数加权

        # 平均 MAE 和平均 loss
        MAE_loss = total_mae / total_samples
        avg_loss = total_loss / total_samples

    return MAE_loss, avg_loss

def test(loader, model, agent):
    total_mae = 0.0
    total_samples = 0
    total_loss = 0.0
    agent._eval()
    model.eval()
    with torch.no_grad():
        for graph in loader:
            graph = graph
            graph = graph.cuda("cuda:0")
            predicts, loss = model(graph, agent)
            mae = torch.abs(predicts.view(-1) - graph.y.view(-1))
            total_mae += mae.sum().item()
            total_samples += graph.y.size(0)

            # 累计总 loss
            total_loss += loss.item() * graph.y.size(0)  # 按样本数加权

            # 平均 MAE 和平均 loss
        MAE_loss = total_mae / total_samples
        avg_loss = total_loss / total_samples

    return MAE_loss, avg_loss

def sub_test(loader, model, agent):
    correct, total_loss = 0, 0
    agent._eval()
    model.eval()
    with torch.no_grad():
        for graph in loader:
            graph = graph.cuda("cuda:0")
            predicts, loss = model(graph, agent)
            # loss = F.nll_loss(predicts, graph.y.view(-1))
            correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
            total_loss += loss.item() * num_graphs(graph)
        test_acc = correct / len(loader.dataset)
        test_loss = total_loss / len(loader.dataset)

    return test_acc, test_loss

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)