

import torch
# from network import Network
import copy
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
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
# 版本4 计算临床的血常规的真实表示和原模型的真实表示之间的距离进行分类
def pretrain(model, epoch_pretrain, batch_size, optimizer, views, train_views, train_label, device):
    model.train()
    train_dataset = TrainDataset(train_views, train_label)
    batch_sampler = Data_Sampler(train_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    # print('train_loader',train_loader)
    t_progress = tqdm(range(args.pretrain_epochs), desc='Pretraining')
    for epoch in t_progress:
        # print('epoch',epoch)
        loss_fn = torch.nn.MSELoss()
        for batch_idx, (xs, y_label) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)
            optimizer_pretrain.zero_grad()
            zs, xrs = model(xs)
            # print('xs[1].shape',xs[1].shape)
            loss_list = []
            for v in range(args.V):
                loss_value = loss_fn(xs[v], xrs[v])
                loss_list.append(loss_value)
            loss = sum(loss_list)
            # print('loss',loss)
            loss.backward()
            optimizer_pretrain.step()


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
    data_para = my_data_dic['hos_160_4view_3c_simple']
    print(data_para)
    txt_file = "./results_con/" + data_para[1] + ".txt"
    f = open(txt_file, 'w').close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    align_epochs = 50
    lr_pre = 0.0005   # 0.0005 预训练学习率
    lr_align = 0.0001  # 0.0001 对齐学习率
    Batch = 128  # 256  batch_size
    Batch2 = 1024
    para_loss = [1e-3, 1e-3]  # 超参数
    feature_dim = 32
    opt_weight_decay = 0.0
    pre_epochs = 200 # 200 预训练epoch
    view_dims=[]
    view_ceshi_dims = []
    seed = 2  # 改
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--dataset', default=data_para)
    parser.add_argument('--batch_size', default=Batch, type=int)
    parser.add_argument('--batch_size_ceshi', default=Batch2, type=int)
    parser.add_argument('--lr_pre', default=lr_pre, type=float)
    parser.add_argument('--lr_align', default=lr_align, type=float)
    parser.add_argument('--pretrain_epochs', default=pre_epochs, type=int)
    parser.add_argument('--align_epochs', default=align_epochs, type=int)
    parser.add_argument("--feature_dim", default=feature_dim)
    parser.add_argument("--weight_decay", default=opt_weight_decay)
    parser.add_argument("--V", default=1)
    parser.add_argument("--K", default=2)
    parser.add_argument("--N", default=160)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--temperature_c", default=0.5)
    parser.add_argument("--temperature_l", default=0.5)
    parser.add_argument("--view_dims", default=view_dims)
    parser.add_argument("--forward_expansion", default=4)
    parser.add_argument("--layers", default=3)
    parser.add_argument("--heads", default=4)
    args = parser.parse_args()

    df = pd.read_excel('1.xlsx')  # 假设数据集存储在Excel文件中
    y = df.iloc[:, -1].values  # 假设标签信息保存在最后一列中
    print('y',y)

    # 提取特征数据

    folder_path = './time_xue/合/verify/'
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    # print(excel_files)
    # 对每个excel文件进行批处理
    for file in excel_files:
        # view_dims = []
        H = []
        H_CESHI = []
        X = []
        X_CESHI = []
        print(file)
        # 读取Excel文件
        df = pd.read_excel(os.path.join(folder_path, file))
        df_large = pd.read_excel('25biaoxing_completed - rename.xlsx')
        # df = pd.read_excel('C:/Users/A/Desktop/DM/病后/verify_final.xlsx')
        # df = pd.read_excel('C:/Users/A/Desktop/DM/revise/verify_compelete_revise.xlsx')
        df_small = df.fillna(0)
        df_small.replace([np.inf, -np.inf], 0, inplace=True)
        df_small = df_small.replace('.', '0')
        df_small = df_small.replace('----', '0')
        df_small = df_small.replace('*0000', '0')
    # verify_features = df.iloc[:, 1:].values
    # df = pd.read_excel('verify_final.xlsx')
    # df = pd.read_excel('C:/Users/A/Desktop/北京/301/9yuexinpaojiyin/25 biaoxing/verify_25/25biaoxing_completed.xlsx')
    # df = pd.read_excel('C:/Users/A/Desktop/DM/revise/verify_compelete_revise.xlsx')
    # df_small = pd.read_excel('C:/Users/A/Desktop/DM/病后/verify_final.xlsx')
    # 提取特征数据
        larger_columns = df_large.columns.tolist()
        smaller_columns  = df_small.columns.tolist()
        print(smaller_columns)
        duplicate_columns = [col for col in larger_columns if col in smaller_columns]

        biaoxing_completed = df_large[duplicate_columns].values
        verify_features = df_small[duplicate_columns].values
        # biaoxing_completed = df_large.values
        # verify_features = df_small.values

        verify_features = np.array(verify_features, dtype=np.float32)
        biaoxing_completed = np.array(biaoxing_completed, dtype=np.float32)
        print('biaoxing_completed',verify_features.shape)
        print('verify_features', biaoxing_completed.shape)
        mm = MinMaxScaler()
        features_scaled = mm.fit_transform(verify_features)
        biaoxing_completed_scaled = mm.fit_transform(biaoxing_completed)

        X.append(biaoxing_completed_scaled)
        X_CESHI.append(features_scaled)
        X_raw = copy.deepcopy(X)
        X_CESHI_raw = copy.deepcopy(X_CESHI)

        X_raw_biao=X[0]
        fea_emb_ceshi = X_CESHI[0]
        df_y = pd.read_excel('y_new.xlsx')
        labels = df_y.iloc[:, -1].values
        print('label',labels)
        raw_label='./time_xue/合/raw_label/output_{}.xlsx'.format(file)
        df = pd.DataFrame({'Cluster Labels': labels})
        df.to_excel(raw_label, index=False)
        k = np.max(labels) + 1  # 簇的数量
        cluster_centers = []
        # cluster_centers_pca = []
        for i in range(k):
            cluster_points = X_raw_biao[labels == i]
            # print(cluster_points)
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
        cluster_centers = np.array(cluster_centers)
        print(cluster_centers)
        pca = PCA(n_components=3)
        X_raw_biao_pca = pca.fit_transform(X_raw_biao)
        cluster_centers_pca = pca.transform(cluster_centers)
        colors = ['g', 'b', 'r']

        # 可视化簇中心和簇结构
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制簇结构
        for i in range(k):
            cluster_points = X_raw_biao_pca[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i + 1}',
                       marker='o', color=colors[i])

        # 绘制簇中心
        for i in range(k):
            ax.scatter(cluster_centers_pca[i, 0], cluster_centers_pca[i, 1], cluster_centers_pca[i, 2], marker='*', s=200,
                       color=colors[i], label=f'Cluster {i + 1} Center')

        for i in range(len(cluster_centers)):
            cluster_indices = np.where(labels == i)[0]
            centroid = cluster_centers[i]
            distances = np.linalg.norm(X_raw_biao[cluster_indices] - centroid, axis=1)
            threshold = np.percentile(distances, 70)
            selected_indices = cluster_indices[np.where(distances <= threshold)[0]]
            for index in selected_indices:
                # ax.scatter(X[index, 0], X[index, 1], X[index, 2], c='red', edgecolors='black')
                ax.plot([X_raw_biao_pca[index, 0], cluster_centers_pca[i][0]], [X_raw_biao_pca[index, 1], cluster_centers_pca[i][1]], [X_raw_biao_pca[index, 2], cluster_centers_pca[i][2]], 'k--')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.set_title('Cluster Visualization')
        ax.legend()
        # plt.show()
        plt.clf()
        features_scaled = fea_emb_ceshi
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_scaled)
        # 计算每个样本到已知中心的距离
        known_centers = cluster_centers  # 已知中心的坐标
        distances = np.zeros((len(features_scaled), len(known_centers)))
        for i, center in enumerate(known_centers):
            distances[:, i] = np.linalg.norm(features_scaled - center, axis=1)

        # 将样本分到距离最近的类中
        labels = np.argmin(distances, axis=1)

        verify_label = './time_xue/合/verify_label/{}'.format(file)
        df = pd.DataFrame({'Cluster Labels': labels})
        df.to_excel(verify_label, index=False)

        # 定义簇中心的颜色
        colors = ['#bbd1e7', '#f3e19c', '#e9b6be']
        colors2 = ['#7c96b3', '#b2a171', '#ae7d8a']
        X_verify_pca = features_pca
        # print('21313',X.shape)
        # 可视化簇中心和簇结构
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制簇结构
        for i in range(k):
            cluster_points = X_verify_pca[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i + 1}',
                       marker='o', color=colors[i], s=20, alpha=0.8)



        for i in range(len(cluster_centers)):
            cluster_indices = np.where(labels == i)[0]
            print('cluster_indices',cluster_indices)
            centroid = cluster_centers[i]
            distances = np.linalg.norm(features_scaled[cluster_indices] - centroid, axis=1)
            for index in cluster_indices:
                ax.plot([X_verify_pca[index, 0], cluster_centers_pca[i][0]], [X_verify_pca[index, 1], cluster_centers_pca[i][1]], [X_verify_pca[index, 2], cluster_centers_pca[i][2]], 'k--',color='#979797', linewidth=0.5)

        # 绘制簇中心
        for i in range(k):
            ax.scatter(cluster_centers_pca[i, 0], cluster_centers_pca[i, 1], cluster_centers_pca[i, 2], marker='*', s=200,
                       color=colors2[i], label=f'Cluster {i + 1} Center')
        ax.legend(fontsize=10,loc='upper right')

        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_zlabel('Feature 3', fontsize=12)
        ax.set_title('Cluster Visualization', fontsize=14)
        # plt.show()
        folder_save_path = './time_xue/合/late_new'
        file_name = 'output_{}.png'.format(file)
        file_name_pdf = 'output_{}.pdf'.format(file)
        # plt.savefig(folder_save_path + '/' + file_name)
        plt.savefig(folder_save_path + '/' + file_name, dpi=1000, bbox_inches='tight')
        plt.savefig(folder_save_path + '/' + file_name_pdf, bbox_inches='tight', format='pdf')
        plt.clf()
