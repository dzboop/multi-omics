import torch
# from network import Network
from Nmetrics import evaluate,get_y_preds

# from torch.utils.data import Dataset
import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score
import argparse
import random
from loss import Loss
import time
# import umap
import torch.nn.functional as F
import load_data as loader
from network import Network
from datasets import Data_Sampler, TrainDataset
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import NormalizeFeaTorch, get_Similarity, clustering, euclidean_dist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
# 多组学学习
def pretrain(model, epoch_pretrain, batch_size, optimizer, views, train_views, train_label, device):
    model.train()
    train_dataset = TrainDataset(train_views, train_label)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

    t_progress = tqdm(range(args.pretrain_epochs), desc='Pretraining')
    for epoch in t_progress:
        loss_fn = torch.nn.MSELoss()
        for batch_idx, (xs, y_label) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            optimizer_pretrain.zero_grad()
            zs, xrs = model(xs)
            loss_list = []
            for v in range(args.V):
                loss_value = loss_fn(xs[v], xrs[v])
                loss_list.append(loss_value)
            loss = sum(loss_list)
            loss.backward()
            optimizer_pretrain.step()
    '''
    fea_emb = []
    for v in range(args.V):
        fea_emb.append([])

    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(train_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_emb[v] = fea_emb[v] + zs2[v].tolist()
    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])'''




def pretrain2(model, epoch_pretrain, batch_size, optimizer, views, train_views, train_label, device):
    model.train()
    train_dataset = TrainDataset(train_views, train_label)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    loss_final = []
    t_progress = tqdm(range(args.align_epochs), desc='Pretraining2')
    for epoch in t_progress:
        total_loss = 0
        loss_fn = torch.nn.MSELoss()
        loss_clf = torch.nn.CrossEntropyLoss()
        for batch_idx, (xs, y_label) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            label = torch.squeeze(y_label[0].long()).to(device)
            optimizer_pretrain.zero_grad()
            zs, xrs = model(xs)
            loss_list1 = []
            loss_list2 = []
            for v in range(args.V):
                loss_value = loss_fn(xs[v], xrs[v])
                loss_clf_value = loss_clf(zs[v], label)
                loss_list1.append(loss_value)
                loss_list2.append(loss_clf_value)
            loss = sum(loss_list1)+sum(loss_list2)
            loss.backward()
            total_loss += loss
            optimizer_pretrain.step()
        loss_final.append(total_loss)
    return loss_final

    '''
    fea_emb = []
    for v in range(args.V):
        fea_emb.append([])

    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(train_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_emb[v] = fea_emb[v] + zs2[v].tolist()
    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])'''

def no_grad_generate(train_loader):
    fea_emb = []
    for v in range(args.V):
        fea_emb.append([])

    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(train_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_emb[v] = fea_emb[v] + zs2[v].tolist()
    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])
    return fea_emb


def train_Con(model, align_epochs, batch_size, optimizer_surprise, views,train_views, train_label, device):

    model.train()
    train_dataset = TrainDataset(train_views, train_label)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    loss_list_final = []
    total_loss = 0
    t_progress = tqdm(range(args.align_epochs), desc='aligning')
    for epoch in t_progress:
        loss_fn = torch.nn.MSELoss()
        loss_clf = torch.nn.CrossEntropyLoss()
        for batch_idx, (xs, y_label) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            label = torch.squeeze(y_label[0].long()).to(device)
            optimizer_surprise.zero_grad()
            zs, xrs = model(xs)
            loss_list = []
            criterion = Loss(zs[0].shape[0], 0.5, 0.5,
                             0.5).cuda()
            for v in range(views):
                # loss_list.append(criterion.contrastive_loss(zs[v][0], z_com))
                for w in range(v + 1, views):
                    loss_list.append(criterion.contrastive_loss(zs[v], zs[w]))
                    # loss_list.append(criterion.forward_label(zs[v][0], zs[w][0]))
                # loss_value2 = loss_fn(z_cats[v], z_com)
                loss_clf_value = loss_clf(zs[v], label)
                loss_value = loss_fn(xs[v], xrs[v])
                loss_list.append(loss_value)
                loss_list.append(loss_clf_value)
                # loss_list.append(loss_value2)
            loss = sum(loss_list)
            loss_list_final.append(loss)
            loss.backward()
            optimizer_surprise.step()

    # fea_emb = []
    # for v in range(args.V):
    #     fea_emb.append([])
    #
    # all_dataset = TrainDataset(X, Y)
    # batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    # all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler_all)
    #
    # with torch.no_grad():
    #     for batch_idx2, (xs2, _) in enumerate(all_loader):
    #         for v in range(args.V):
    #             xs2[v] = torch.squeeze(xs2[v]).to(device)
    #         zs2, xrs2 = model(xs2)
    #         for v in range(args.V):
    #             zs2[v] = zs2[v].cpu()
    #             fea_emb[v] = fea_emb[v] + zs2[v].tolist()
    # for v in range(args.V):
    #     fea_emb[v] = torch.tensor(fea_emb[v])
    #
    # return fea_emb


def spectral_clustering(features, n_clusters):
    # 构建相似度矩阵
    # similarity_matrix = np.exp(-0.5 * np.linalg.norm(features[:, np.newaxis] - features, axis=2) ** 2)

    # 执行谱聚类
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=1)
    labels = sc.fit_predict(features)

    return labels


def k_emeans(fea_emb,K,y_true):
    #输入表示输出聚类结果
    kmeans = KMeans(n_clusters=K)
    #kmeans.fit(fea_emb.numpy())
    kmeans.fit(fea_emb)
    pred2 = kmeans.labels_
    # print(pred2)
    acc, nmi, purity, fscore, precision, recall, ari = evaluate(y_true, pred2)
    print('ACC=%.4f, NMI=%.4f, PUR=%.4f, Fscore=%.4f, Prec=%.4f, Recall=%.4f, ARI=%.4f' %
          (acc, nmi, purity, fscore, precision, recall, ari))
    my_dic = dict({'ACC': acc, 'NMI': nmi, 'PUR': purity,
                   'Fscore': fscore, 'Prec': precision, 'recall': recall, 'ARI': ari})  # 改
    writers(my_dic)

def writers(my_dic2):
    with open(txt_file, 'r+', newline='', encoding='utf-8') as f1:
        content = f1.read()
        f1.seek(0, 2)
        f1.write(str(my_dic2) + '\r')  # 没有'\r'不会换行，会接着原来的内容

################## main ##################
my_data_dic = loader.ALL_data
for _ in range(1):
    data_para = my_data_dic['hos_160_4view_3c_simple']
    print(data_para)
    txt_file = "./results_con/" + data_para[1] + ".txt"
    f = open(txt_file, 'w').close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    align_epochs = 50
    lr_pre = 0.0005    # 0.0005 预训练学习率
    lr_align = 0.0001  # 0.0001 对齐学习率
    Batch = 16  # 256  batch_size
    para_loss = [1e-3, 1e-3]  # 超参数
    feature_dim = 128 # 128
    opt_weight_decay = 0.0
    pre_epochs = 200  # 200 预训练epoch

    seed = 1  # 改
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--dataset', default=data_para)
    parser.add_argument('--batch_size', default=Batch, type=int)
    parser.add_argument('--lr_pre', default=lr_pre, type=float)
    parser.add_argument('--lr_align', default=lr_align, type=float)
    parser.add_argument('--pretrain_epochs', default=pre_epochs, type=int)
    parser.add_argument('--align_epochs', default=align_epochs, type=int)
    parser.add_argument("--feature_dim", default=feature_dim)
    parser.add_argument("--weight_decay", default=opt_weight_decay)
    parser.add_argument("--V", default=data_para['V'])
    parser.add_argument("--K", default=data_para['K'])
    parser.add_argument("--N", default=data_para['N'])
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--temperature_c", default=0.5)
    parser.add_argument("--temperature_l", default=0.5)
    parser.add_argument("--view_dims", default=data_para['n_input'])
    parser.add_argument("--forward_expansion", default=4)
    parser.add_argument("--layers", default=3)
    parser.add_argument("--heads", default=4)
    args = parser.parse_args()

    X, Y = loader.load_data(args.dataset)


    train_ratio = 0.8
    test_ratio = 0.2
    # 使用列表推导式将四个视图的数据分别切分成训练数据和测试数据
    train_views = [view[:int(len(view) * train_ratio)] for view in X]
    test_views = [view[int(len(view) * train_ratio):] for view in X]
    train_label = [view[:int(len(view) * train_ratio)] for view in Y]
    test_label = [view[int(len(view) * train_ratio):] for view in Y]

    ###### 对raw features进行kmeans ######
    kmeans = KMeans(n_clusters=args.K)
    y_true2 = Y[0]
    # print(y_true2)
    print('Raw Features--')
    my_dic = 'Raw Features--'
    writers(my_dic)
    for v in range(args.V):
        # print(X[v].shape)
        k_emeans(X[v],args.K,y_true2)

    model = Network(args.V, args.view_dims, args.feature_dim, args.layers, args.heads, args.forward_expansion).to(device)
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=args.lr_pre, weight_decay=args.weight_decay)
    pretrain(model, args.pretrain_epochs, args.batch_size, optimizer_pretrain, args.V, train_views, train_label,
                       device)  # fea_all 为各视图经过AE得到的nxd维的特征
    with torch.no_grad():
        for v in range(args.V):
            train_views[v] = torch.squeeze(train_views[v]).to(device)
        zs2, xrs2 = model(train_views)
        for v in range(args.V):
            zs2[v] = zs2[v].cpu()
    fea_emb=zs2
    for v in range(args.V):
        k_emeans(fea_emb[v], args.K, train_label[0])

    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])
    optimizer_surprise = torch.optim.Adam(model.parameters(), lr=args.lr_align, weight_decay=args.weight_decay)
    loss_final = pretrain2(model, args.align_epochs, args.batch_size, optimizer_surprise, args.V, train_views, train_label,
                       device)  # fea_all 为各视图经过AE得到的nxd维的特征
    # loss_final_np = [loss.detach().cpu().numpy() for loss in loss_final]
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(loss_final_np , marker='o', color='#bbd1e7', label='Training Loss')
    # # 添加标题和坐标轴标签
    # plt.title('Loss Convergence Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # # 添加图例
    # plt.legend()
    # # 添加网格线（可选）
    # plt.grid(True)
    # # 显示图表
    # plt.savefig('loss-epoch.png', dpi=1000, bbox_inches='tight')
    # plt.savefig('loss-epoch.pdf', bbox_inches='tight', format='pdf')
    # plt.show()

    with torch.no_grad():
        for v in range(args.V):
            test_views[v] = torch.squeeze(test_views[v]).to(device)
        zs2, xrs2 = model(test_views)
        for v in range(args.V):
            fea_emb[v] = zs2[v].cpu()
    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])
    y_true = test_label[0]
    fea_fina1 = torch.cat(fea_emb, dim=1)
    print(y_true)
    k_emeans(fea_fina1, args.K, y_true)
    # for v in range(args.V):
    #     k_emeans(fea_emb[v], args.K, y_true)


    # ==================================================================================
    print('=====================================================================')
    my_dic = '====================================================================='
    writers(my_dic)
    my_dic = 'on representations:'
    writers(my_dic)
    print('on representations:')
    # =====================================find pathway============================================
    # with torch.no_grad():
    #     for v in range(args.V):
    #         X[v] = torch.squeeze(X[v]).to(device)
    #     zs2, xrs2 = model(X)
    #     for v in range(args.V):
    #         fea_emb[v] = zs2[v].cpu()
    # for v in range(args.V):
    #     fea_emb[v] = torch.tensor(fea_emb[v])
    # print('after cat')
    all_dataset = TrainDataset(X, Y)
    batch_sampler = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler)
    fea_emb = no_grad_generate(all_loader)

    fea_fina_x = torch.cat(fea_emb,dim=1)
    # fea_fina_x = sum(fea_emb)
    kmeans = KMeans(n_clusters=args.K)
    kmeans.fit(fea_fina_x)
    pred = kmeans.labels_
    y_pred_ajusted = get_y_preds(y_true2, pred, args.K)
    final_acc = f1_score(y_true2, y_pred_ajusted,average='weighted')
    print("final_acc",final_acc)
    # X_clone = copy.deepcopy(X)
    # # test_views_clone = copy.deepcopy(test_views)
    # X_imp = [[0 for i in range(args.view_dims[v])] for v in range(args.V)]
    # for v in range(args.V):
    #     print('View %d Selection...' % (v + 1))
    #     for i in range(args.view_dims[v]):
    #         X = copy.deepcopy(X_clone)
    #         X[v][:, i] = 0
    #         #
    #         # with torch.no_grad():
    #         #     for view1 in range(args.V):
    #         #         X[view1] = torch.squeeze(X[view1]).to(device)
    #         #     zs2, xrs2 = model(X)
    #         #     for view2 in range(args.V):
    #         #         fea_emb[view2] = zs2[view2].cpu()
    #         # for view3 in range(args.V):
    #         #     fea_emb[view3] = torch.tensor(fea_emb[view3])
    #         train_dataset = TrainDataset(X, Y)
    #         batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    #         train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    #         fea_emb = no_grad_generate(train_loader)
    #         adjust_emb = torch.cat(fea_emb,dim=1)
    #         # adjust_emb = sum(fea_emb)
    #         kmeans = KMeans(n_clusters=args.K)
    #         kmeans.fit(adjust_emb)
    #         pred2 = kmeans.labels_
    #         y_pred_ajusted2 = get_y_preds(y_true2, pred2, args.K)
    #         scores_1 = f1_score(y_true2, y_pred_ajusted2,average='weighted')
    #         # print("scores_1", scores_1)
    #         X_imp[v][i] = (final_acc - scores_1) * 100
    #     sorted_lst = sorted(enumerate(X_imp[v]), key=lambda x: x[1], reverse=True)
    #
    #     indexes = list(map(lambda x: x[0], sorted_lst[:200]))
    #     values = list(map(lambda x: x[1], sorted_lst[:200]))
    #
    #     print('View %d' % (v + 1))
    #     print("Indexes:", indexes)
    #     print("Values:", values)

    '''
    optimizer_align = torch.optim.Adam(model.parameters(), lr=args.lr_align, weight_decay=args.weight_decay)
    fea_final = train_Con(model, args.align_epochs,args.batch_size,optimizer_align, args.V,X, device)
    kmeans = KMeans(n_clusters=args.K)
    y_true = Y[0]
    my_dic = 'on after con representations:'
    writers(my_dic)
    my_dic = '====================================================================='
    writers(my_dic)
    print('=====================================================================')
    print('on after con representations:')
    for v in range(args.V):
        k_emeans(fea_final[v], args.K, y_true)
    my_dic = 'on after con and cat representations:'
    writers(my_dic)
    fea_fina2 = torch.cat(fea_final, dim=1)
    print('on after con and cat representations:')
    k_emeans(fea_fina2, args.K, y_true)

    # ==================================================================================
    print('=====================================================================')
    '''