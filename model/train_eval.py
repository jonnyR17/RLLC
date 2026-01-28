import torch


def _num_graphs(batch):
    if hasattr(batch, 'num_graphs') and batch.num_graphs is not None:
        return int(batch.num_graphs)
    if hasattr(batch, 'batch') and batch.batch is not None:
        return int(batch.batch.max().item() + 1)
    return 1


def train(loader, model, agent, optimizer, device=None):
    if device is None:
        device = next(model.parameters()).device

    correct, total_loss = 0, 0
    agent._eval()       # 训练GNN阶段：agent固定
    model.train()

    for graph in loader:
        graph = graph.to(device)
        optimizer.zero_grad()

        predicts, loss = model(graph, agent)   # forward 返回 (predicts, loss)
        loss.backward()
        optimizer.step()

        pred = predicts.argmax(dim=1)
        correct += pred.eq(graph.y.view(-1)).sum().item()
        total_loss += loss.item() * _num_graphs(graph)

    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return acc, avg_loss


@torch.no_grad()
def test(loader, model, agent, device=None):
    if device is None:
        device = next(model.parameters()).device

    correct, total_loss = 0, 0
    agent._eval()
    model.eval()

    for graph in loader:
        graph = graph.to(device)
        predicts, loss = model(graph, agent)

        pred = predicts.argmax(dim=1)
        correct += pred.eq(graph.y.view(-1)).sum().item()
        total_loss += loss.item() * _num_graphs(graph)

    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return acc, avg_loss



# import time
#
# from torch_geometric.utils import *
# import networkx as nx
# import torch
# import torch.nn.functional as F
# from torch import tensor
# from torch.optim import Adam
# from sklearn.model_selection import StratifiedKFold
# import matplotlib.pyplot as plt
# from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
# def learn(loader, model, agent, optimizer, recorder=None):
#     """
#
#     Args:
#         loader: train_loader=DenseLoader(train_dataset,batch_size)
#         model: NET=model.Sugar
#         agent: AgentChain
#         optimizer: Adam
#         recorder: None
#
#     Returns:
#         learn_acc,learn_loss
#     """
#     correct, total_loss,rewards = 0, 0, 0
#     agent._train()#AgentChain._train()
#     model.train()#Sugar.forward()
#     for graph in loader:
#         graph = graph
#         graph = graph.cuda("cuda:0")
#         optimizer.zero_grad()
#         predicts, loss = model(graph, agent)
#         loss.backward(retain_graph=True)
#         optimizer.step()
#
#         correct_vector = predicts.max(1)[1].eq(graph.y.view(-1))
#         correct += correct_vector.sum().item()
#         total_loss += loss.item() * num_graphs(graph)
#
#     train_acc = correct / len(loader.dataset)
#     train_loss = total_loss / len(loader.dataset)
#     return train_acc, train_loss
#
# # def train(loader, model, agent, optimizer):
# #
# #     """
# #      Args:
# #          loader: train_loader=DenseLoader(train_dataset,batch_size)
# #          model: NET=model.Sugar
# #          agent: AgentChain
# #          optimizer: Adam
# #
# #      Returns:
# #          train_acc,train_loss
# #      """
# #     correct, total_loss = 0, 0
# #     agent._eval()
# #     model.train()
# #     for graph in loader:
# #         graph = graph.cuda("cuda:0")
# #         optimizer.zero_grad()
# #         predicts,  loss = model(graph, agent) #Net 网络由两个卷积与两层线性与一个区分器构成
# #         # loss = F.nll_loss(predicts, graph.y.view(-1))
# #         loss.backward()
# #         # correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
# #         correct_vector = predicts.max(1)[1].eq(graph.y.view(-1))
# #         # agent.fed_reward(correct_vector)
# #
# #         correct += correct_vector.sum().item()
# #         total_loss += loss.item() * num_graphs(graph)
# #         optimizer.step()
# #     train_acc = correct / len(loader.dataset)
# #     train_loss = total_loss / len(loader.dataset)
# #
# #     return train_acc, train_loss
# #
# # def eval(loader, model, agent):
# #     correct, total_loss = 0, 0
# #     agent._eval()
# #     model.eval()
# #     with torch.no_grad():
# #         for graph in loader:
# #             graph = graph.cuda("cuda:0")
# #             predicts,  loss = model(graph, agent)
# #             # loss = F.nll_loss(predicts, graph.y.view(-1))
# #             correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
# #             total_loss += loss.item() * num_graphs(graph)
# #         eval_acc = correct / len(loader.dataset)
# #         eval_loss = total_loss / len(loader.dataset)
# #     return eval_acc, eval_loss
# #
# # def test(loader, model, agent):
# #     correct, total_loss = 0, 0
# #     agent._eval()
# #     model.eval()
# #     with torch.no_grad():
# #         for graph in loader:
# #             graph = graph.cuda("cuda:0")
# #             predicts, loss= model(graph, agent)
# #             # loss = F.nll_loss(predicts, graph.y.view(-1))
# #             correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
# #             total_loss += loss.item() * num_graphs(graph)
# #         test_acc = correct / len(loader.dataset)
# #         test_loss = total_loss / len(loader.dataset)
# #     return test_acc, test_loss
# def train(loader, model, agent, optimizer):
#
#     """
#      Args:
#          loader: train_loader=DenseLoader(train_dataset,batch_size)
#          model: NET=model.Sugar
#          optimizer: Adam(model.parameters,lr)
#          agent: a reinforcement learning agent.
#      Returns:
#         loss:
#     """
#     total_loss, correct = 0, 0
#     agent._train()
#     model.train()
#     device = next(model.parameters()).device
#
#     for graph in loader:
#         graph = graph.to(device)
#         optimizer.zero_grad()
#         predicts, loss = model(graph, agent)
#         # loss = loss_label + loss_sub + loss_mi
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     return total_loss / len(loader)
#
#
# def eval(loader, model, agent):
#     correct, total_loss = 0, 0
#     agent._eval()
#     model.eval()
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         for graph in loader:
#             graph = graph.to(device)
#             predict = model(graph, agent)
#             loss = F.nll_loss(predict, graph.y.view(-1))
#             total_loss += loss.item()
#
#             pred = predict.max(dim=1)[1]
#             correct += pred.eq(graph.y.view(-1)).sum().item()
#     return total_loss / len(loader), correct / len(loader.dataset)
#
#
# def test(loader, model, agent):
#     correct, total_loss = 0, 0
#     agent._eval()
#     model.eval()
#     device = next(model.parameters()).device
#
#     with torch.no_grad():
#         for graph in loader:
#             graph = graph.to(device)
#
#             out = model(graph, agent)
#
#             # Sugar.forward() 默认返回 (predicts, loss)
#             if isinstance(out, (tuple, list)):
#                 predicts, loss = out[0], out[1]
#             else:
#                 predicts = out
#                 loss = F.nll_loss(predicts, graph.y.view(-1))
#
#             correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
#             total_loss += loss.item() * num_graphs(graph)
#
#     test_acc = correct / len(loader.dataset)
#     test_loss = total_loss / len(loader.dataset)
#     return test_acc, test_loss
#
#
# def get_embedding(loader, model, agent):
#     agent._eval()
#     model.eval()
#     device = next(model.parameters()).device
#     graph_embedding_list, graph_label_list = [], []
#     with torch.no_grad():
#         for graph in loader:
#             graph = graph.to(device)
#             graph_embedding, graph_label = model.get_graph_embedding(graph, agent)
#             graph_embedding_list.append(graph_embedding)
#             graph_label_list.append(graph_label)
#     graph_embedding_list = torch.cat(graph_embedding_list)
#     graph_label_list = torch.cat(graph_label_list)
#     return graph_embedding_list, graph_label_list
#
# def sub_test(loader, model, agent):
#     correct, total_loss = 0, 0
#     agent._eval()
#     model.eval()
#     with torch.no_grad():
#         for graph in loader:
#             graph = graph.cuda("cuda:0")
#             predicts, loss = model(graph, agent)
#             # loss = F.nll_loss(predicts, graph.y.view(-1))
#             correct += predicts.max(1)[1].eq(graph.y.view(-1)).sum().item()
#             total_loss += loss.item() * num_graphs(graph)
#         test_acc = correct / len(loader.dataset)
#         test_loss = total_loss / len(loader.dataset)
#
#     return test_acc, test_loss
#
# def num_graphs(data):
#     if hasattr(data, 'num_graphs'):
#         return data.num_graphs
#     else:
#         return data.x.size(0)