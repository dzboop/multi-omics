import torch
# from network import Network
from Nmetrics import evaluate
# from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
from sklearn.metrics import f1_score
import argparse
import random
from loss import Loss
import time
# import umap
import pandas as pd
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
# 生成原始标签
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

    t_progress = tqdm(range(args.align_epochs), desc='Pretraining2')
    for epoch in t_progress:
        loss_fn = torch.nn.MSELoss()
        loss_clf = torch.nn.CrossEntropyLoss()
        for batch_idx, (xs, y_label) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            label = torch.squeeze(y_label[0].long()).to(device)
            optimizer_pretrain.zero_grad()
            zs, xrs = model(xs)
            loss_list = []
            for v in range(args.V):
                loss_value = loss_fn(xs[v], xrs[v])
                loss_clf_value = loss_clf(zs[v], label)
                loss_list.append(loss_value)
                loss_list.append(loss_clf_value)
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


def train_Con(model, align_epochs, batch_size, optimizer, views, X_list, device):
    model.train()
    train_dataset = TrainDataset(X, Y)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)

    t_progress = tqdm(range(args.pretrain_epochs), desc='Pretraining')
    for epoch in t_progress:
        loss_fn = torch.nn.MSELoss()
        for batch_idx, (xs, _) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            optimizer_pretrain.zero_grad()
            zs, xrs = model(xs)
            loss_list = []
            for v in range(views):
                # loss_list.append(criterion.contrastive_loss(zs[v][0], z_com))
                for w in range(v + 1, views):
                    loss_list.append(criterion.contrastive_loss(zs[v], zs[w]))
                #     loss_list.append(criterion.forward_label(zs[v][0], zs[w][0]))
                # loss_value2 = loss_fn(z_cats[v], z_com)
                loss_value = loss_fn(xs[v], xrs[v])
                loss_list.append(loss_value)
                # loss_list.append(loss_value2)
            loss = sum(loss_list)
            loss.backward()
            optimizer_pretrain.step()

    fea_emb = []
    for v in range(args.V):
        fea_emb.append([])

    all_dataset = TrainDataset(X, Y)
    batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler_all)

    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(all_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_emb[v] = fea_emb[v] + zs2[v].tolist()
    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])

    return fea_emb


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
    print(pred2)
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
    data_para = my_data_dic['hos_160_5view_3c_25_simple']
    print(data_para)
    txt_file = "./results_con/" + data_para[1] + ".txt"
    f = open(txt_file, 'w').close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    align_epochs = 50
    lr_pre = 0.0005    # 0.0005 预训练学习率
    lr_align = 0.0001  # 0.0001 对齐学习率
    Batch = 16  # 256  batch_size
    para_loss = [1e-3, 1e-3]  # 超参数
    feature_dim = 128
    opt_weight_decay = 0.0
    pre_epochs = 100 # 200 预训练epoch

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
    H=[]
    X, Y = loader.load_data(args.dataset)
    df = pd.read_excel('1.xlsx')  # 假设数据集存储在Excel文件中
    y = df.iloc[:, -1].values  # 假设标签信息保存在最后一列中

    for i in range(args.V):
        H.append(y)

    ###### 对raw features进行kmeans ######
    kmeans = KMeans(n_clusters=args.K)
    y_true2 = y

    print('Raw Features--')
    my_dic = 'Raw Features--'
    writers(my_dic)
    for v in range(args.V):
        # print(X[v].shape)
        k_emeans(X[v],args.K,y_true2)

    model = Network(args.V, args.view_dims, args.feature_dim, args.layers, args.heads, args.forward_expansion).to(device)
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=args.lr_pre, weight_decay=args.weight_decay)
    pretrain(model, args.pretrain_epochs, args.batch_size, optimizer_pretrain, args.V, X, H,
                       device)  # fea_all 为各视图经过AE得到的nxd维的特征
    with torch.no_grad():
        for v in range(args.V):
            X[v] = torch.squeeze(X[v]).to(device)
        zs2, xrs2 = model(X)
        for v in range(args.V):
            zs2[v] = zs2[v].cpu()
    fea_emb=zs2
    for v in range(args.V):
        k_emeans(fea_emb[v], args.K, y)
    X=fea_emb[4]
    print(X.shape)
    X = X.numpy()
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    pred2 = kmeans.labels_
    # for v in range(args.V):
    #     fea_emb[v] = torch.tensor(fea_emb[v])
    labels = pred2
    print(labels)

    k = np.max(labels) + 1  # 簇的数量
    cluster_centers = []
    cluster_centers_pca = []
    for i in range(k):
        cluster_points = X[labels == i]
        # print(cluster_points)
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)
    # print(cluster_centers)
    pca = PCA(n_components=3)
    X = pca.fit_transform(X)
    cluster_centers = pca.transform(cluster_centers)
    print(cluster_centers)
    # 定义簇中心的颜色
    # colors = ['g', 'b', 'r']
    colors = ['#bbd1e7', '#f3e19c', '#e9b6be']
    colors2 = ['#7c96b3', '#b2a171', '#ae7d8a']
    # 7c96b3
    # 可视化簇中心和簇结构
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制簇结构
    for i in range(k):
        cluster_points = X[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i + 1}',
                   marker='o', color=colors[i], s=20, alpha=0.8)

    # 绘制簇中心
    for i in range(k):
        ax.scatter(cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2], marker='*', s=200,
                   color=colors2[i], label=f'Cluster {i + 1} Center')

    for i in range(len(cluster_centers)):
        cluster_indices = np.where(labels == i)[0]
        centroid = cluster_centers[i]
        distances = np.linalg.norm(X[cluster_indices] - centroid, axis=1)
        threshold = np.percentile(distances, 80)
        selected_indices = cluster_indices[np.where(distances <= threshold)[0]]
        for index in selected_indices:
            ax.plot([X[index, 0], centroid[0]], [X[index, 1], centroid[1]], [X[index, 2], centroid[2]], 'k--',
                    color='#979797',linewidth=0.5)

    # 调整标签的字体大小
    ax.legend(fontsize=10)

    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_zlabel('Feature 3', fontsize=12)
    ax.set_title('Cluster Visualization', fontsize=14)

    # 保存为高清图片
    plt.savefig('dhweui.png', dpi=1000, bbox_inches='tight')
    plt.savefig('dhweui.pdf', bbox_inches='tight', format='pdf')
    plt.show()


