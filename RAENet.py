import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from data.datasets import twin_datasets,ihdp_datasets,twin_datasets_csv,jobs_datasets
from torch.utils.data import DataLoader
from models.IPW import IPW,AIPW,CBPSNet
from models.VAE import VAE,Autoencoder
from models.GAN import Discriminator
from models.subspaceTransformerITE import OutNet
import seaborn as sns
import argparse
import random
import itertools
from sklearn.manifold import TSNE,Isomap,MDS
import matplotlib.pyplot as plt
from utils.metrics import PEHE, ATE,PEHE_IHDP,ATE_IHDP,sqrt_PEHE,abs_error_ATE,abs_ate,pehe_val,ATT,ROL,Jobs_metric
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)
def loss_fn(recon_x, x, mu, log_var):
    x = x.detach()
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
def mainTwins(args):
    # 构造数据集
    train_dataset = twin_datasets(isTrain=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    #Reweighting = AIPW(31,args.hid_dims,30).to(device)
    #Reweighting = IPW(30, args.hid_dims, 30).to(device)
    Reweighting = CBPSNet(30, args.hid_dims, 30).to(device)
    Var_AE = VAE(args.input_dims,args.hid_dims,args.out_dims).to(device)
    # Discriminator
    discriminator = Discriminator(args.input_dims, args.hid_dims, args.out_dims).to(device)
    outnet_t0 = OutNet(args.hid_dims, args.out_dims).to(device)
    outnet_t1 = OutNet(args.hid_dims, args.out_dims).to(device)
    optimizer_G = Adam(itertools.chain(Reweighting.parameters(),Var_AE.parameters(),
                                       outnet_t0.parameters(), outnet_t1.parameters()), lr=args.lr)
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr)

    min_pehe = 99999
    min_ate = 99999
    epoch_list = []
    AIPW_list = []
    IPW_list = []
    CBPS_list = []
    # for epoch in range(args.epoch):
    #     epoch_pehe=9999
    #     epoch_ate=9999
    #     for steps, [train_x, train_t, train_y, train_potential_y] in enumerate(train_dataloader):
    #         train_x = train_x.float().to(device)
    #         train_t = train_t.float().to(device)
    #         train_y = train_y.float().to(device)
    #
    #         input_y0 = train_y[train_t == 0].float().to(device)
    #         input_y1 = train_y[train_t == 1].float().to(device)
    #
    #         Embedding = Reweighting(train_x,train_t.unsqueeze(dim=1))
    #         recon_x, mu, log_var = Var_AE(Embedding)
    #         y_hat_0 = outnet_t0(Embedding[train_t == 0])
    #         y_hat_1 = outnet_t1(Embedding[train_t == 1])
    #
    #         for i in range(args.dis_epoch):
    #             optimizer_D.zero_grad()
    #             # Adversarial loss
    #             loss_D = -torch.mean(discriminator(Embedding[train_t == 0])) + torch.mean(
    #                 discriminator(Embedding[train_t == 1]))
    #             loss_D.backward(retain_graph=True)
    #             optimizer_D.step()
    #
    #             # Clip weights of discriminator
    #             for p in discriminator.parameters():
    #                 p.data.clamp_(-args.clip_value, args.clip_value)
    #
    #         # loss_G
    #         loss_G = -torch.mean(discriminator(Embedding[train_t == 1]))
    #
    #         # loss reconstruction
    #         loss_V = loss_fn(recon_x, Embedding, mu, log_var)
    #         # loss prediction
    #         loss_pred = F.mse_loss(y_hat_0.squeeze(), input_y0).to(device) + F.mse_loss(y_hat_1.squeeze(),
    #                                                                                         input_y1).to(device)
    #         # total loss
    #         loss_other = loss_G + loss_V + args.rescon * loss_pred
    #         optimizer_G.zero_grad()
    #         loss_other.backward()
    #         optimizer_G.step()
    #
    #         if steps % args.print_steps == 0 or steps == 0:
    #             print(
    #                 "Epoches: %d, step: %d, loss_D:%.3f,loss_G:%.3f,loss_V:%.3f,loss_pre:%.3f"
    #                 % (epoch, steps, loss_D.detach().cpu().numpy(), loss_G.detach().cpu().numpy(),
    #                     loss_V.detach().cpu().numpy(), loss_pred.detach().cpu().numpy()))
    #             # ---------------------
    #             #         Test
    #             # ---------------------
    #             Reweighting.eval()
    #             Var_AE.eval()
    #             discriminator.eval()
    #             outnet_t0.eval()
    #             outnet_t1.eval()
    #             total_test_potential_y = torch.Tensor([]).to(device)
    #             total_test_potential_y_hat = torch.Tensor([]).to(device)
    #             test_dataset = twin_datasets(isTrain=False)
    #             test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    #             for steps, [test_x, test_t,test_potential_y] in enumerate(test_dataloader):
    #                 test_x = test_x.float().to(device)
    #                 test_t =test_t.float().to(device)
    #                 test_potential_y = test_potential_y.float().to(device)
    #                 Embedding_test= Reweighting(test_x,test_t.unsqueeze(dim=1))
    #                 recon_x, mu, log_var = Var_AE(Embedding_test)
    #                 y_hat_t0 = outnet_t0(Embedding_test)
    #                 y_hat_t1 = outnet_t1(Embedding_test)
    #                 test_potential_y_hat = torch.cat([y_hat_t0, y_hat_t1], dim=-1)
    #                 total_test_potential_y = torch.cat([total_test_potential_y, test_potential_y], dim=0)
    #                 total_test_potential_y_hat = torch.cat([total_test_potential_y_hat, test_potential_y_hat],
    #                                                            dim=0)
    #
    #             pehe = PEHE(total_test_potential_y_hat.cpu().detach().numpy(),
    #                             total_test_potential_y.cpu().detach().numpy())
    #             ate = ATE(total_test_potential_y_hat.cpu().detach().numpy(),
    #                           total_test_potential_y.cpu().detach().numpy())
    #
    #             print("PEHE:", pehe)
    #             print("ATE:", ate)
    #             epoch_pehe=min(pehe, epoch_pehe)
    #             epoch_ate = min(ate, epoch_ate)
    #             min_pehe = min(pehe, min_pehe)
    #             min_ate = min(ate, min_ate)
    #             Reweighting.train()
    #             Var_AE.train()
    #             discriminator.train()
    #             outnet_t0.train()
    #             outnet_t1.train()
    #             epoch_list.append(epoch)
    #             AIPW_list.append(epoch_pehe)
    #     print("PEHE:", min_pehe)
    #     print("ATE:", min_ate)
    #tsne_reduction_embedding_AIPW = TSNE(n_components=2,perplexity=100, n_iter=5000)
    #reduce_train_embedding_t0 = tsne_reduction_embedding_AIPW.fit_transform(train_embedding_t0.detach().cpu().numpy())
    #reduce_train_embedding_t1 = tsne_reduction_embedding_AIPW.fit_transform(train_embedding_t1.detach().cpu().numpy())
    #plt.scatter(reduce_train_embedding_t0[:, 0], reduce_train_embedding_t0[:, 1], c='red',label="t0")
    #plt.scatter(reduce_train_embedding_t1[:, 0], reduce_train_embedding_t1[:, 1], c='green',label="t1")
    #plt.legend()
    #plt.savefig('AIPW_embedding.png')
    # for epoch in range(args.epoch):
    #     epoch_pehe=9999
    #     epoch_ate=9999
    #     for steps, [train_x, train_t, train_y, train_potential_y] in enumerate(train_dataloader):
    #         train_x = train_x.float().to(device)
    #         train_t = train_t.float().to(device)
    #         train_y = train_y.float().to(device)
    #
    #         input_y0 = train_y[train_t == 0].float().to(device)
    #         input_y1 = train_y[train_t == 1].float().to(device)
    #
    #         Embedding = Reweighting(train_x)
    #         recon_x, mu, log_var = Var_AE(Embedding)
    #         y_hat_0 = outnet_t0(Embedding[train_t == 0])
    #         y_hat_1 = outnet_t1(Embedding[train_t == 1])
    #
    #         for i in range(args.dis_epoch):
    #             optimizer_D.zero_grad()
    #             # Adversarial loss
    #             loss_D = -torch.mean(discriminator(Embedding[train_t == 0])) + torch.mean(
    #                 discriminator(Embedding[train_t == 1]))
    #             loss_D.backward(retain_graph=True)
    #             optimizer_D.step()
    #
    #             # Clip weights of discriminator
    #             for p in discriminator.parameters():
    #                 p.data.clamp_(-args.clip_value, args.clip_value)
    #
    #         # loss_G
    #         loss_G = -torch.mean(discriminator(Embedding[train_t == 1]))
    #
    #         # loss reconstruction
    #         loss_V = loss_fn(recon_x, Embedding, mu, log_var)
    #         # loss prediction
    #         loss_pred = F.mse_loss(y_hat_0.squeeze(), input_y0).to(device) + F.mse_loss(y_hat_1.squeeze(),
    #                                                                                         input_y1).to(device)
    #         # total loss
    #         loss_other = loss_G + loss_V + args.rescon * loss_pred
    #         optimizer_G.zero_grad()
    #         loss_other.backward()
    #         optimizer_G.step()
    #
    #         if steps % args.print_steps == 0 or steps == 0:
    #             print(
    #                 "Epoches: %d, step: %d, loss_D:%.3f,loss_G:%.3f,loss_V:%.3f,loss_pre:%.3f"
    #                 % (epoch, steps, loss_D.detach().cpu().numpy(), loss_G.detach().cpu().numpy(),
    #                     loss_V.detach().cpu().numpy(), loss_pred.detach().cpu().numpy()))
    #             # ---------------------
    #             #         Test
    #             # ---------------------
    #             Reweighting.eval()
    #             Var_AE.eval()
    #             discriminator.eval()
    #             outnet_t0.eval()
    #             outnet_t1.eval()
    #             total_test_potential_y = torch.Tensor([]).to(device)
    #             total_test_potential_y_hat = torch.Tensor([]).to(device)
    #             test_dataset = twin_datasets(isTrain=False)
    #             test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    #             for steps, [test_x, test_t,test_potential_y] in enumerate(test_dataloader):
    #                 test_x = test_x.float().to(device)
    #                 test_t =test_t.float().to(device)
    #                 test_potential_y = test_potential_y.float().to(device)
    #                 Embedding_test= Reweighting(test_x)
    #                 recon_x, mu, log_var = Var_AE(Embedding_test)
    #                 y_hat_t0 = outnet_t0(Embedding_test)
    #                 y_hat_t1 = outnet_t1(Embedding_test)
    #                 test_potential_y_hat = torch.cat([y_hat_t0, y_hat_t1], dim=-1)
    #                 total_test_potential_y = torch.cat([total_test_potential_y, test_potential_y], dim=0)
    #                 total_test_potential_y_hat = torch.cat([total_test_potential_y_hat, test_potential_y_hat],
    #                                                            dim=0)
    #
    #             pehe = PEHE(total_test_potential_y_hat.cpu().detach().numpy(),
    #                             total_test_potential_y.cpu().detach().numpy())
    #             ate = ATE(total_test_potential_y_hat.cpu().detach().numpy(),
    #                           total_test_potential_y.cpu().detach().numpy())
    #
    #             print("PEHE:", pehe)
    #             print("ATE:", ate)
    #             epoch_pehe=min(pehe, epoch_pehe)
    #             epoch_ate = min(ate, epoch_ate)
    #             min_pehe = min(pehe, min_pehe)
    #             min_ate = min(ate, min_ate)
    #             Reweighting.train()
    #             Var_AE.train()
    #             discriminator.train()
    #             outnet_t0.train()
    #             outnet_t1.train()
    #             IPW_list.append(epoch_pehe)
    #     print("PEHE:", min_pehe)
    #     print("ATE:", min_ate)
    for epoch in range(args.epoch):
        epoch_pehe=9999
        epoch_ate=9999
        for steps, [train_x, train_t, train_y, train_potential_y] in enumerate(train_dataloader):
            train_x = train_x.float().to(device)
            train_t = train_t.float().to(device)
            train_y = train_y.float().to(device)

            input_y0 = train_y[train_t == 0].float().to(device)
            input_y1 = train_y[train_t == 1].float().to(device)

            Embedding = Reweighting(train_x)
            recon_x, mu, log_var = Var_AE(Embedding)
            y_hat_0 = outnet_t0(Embedding[train_t == 0])
            y_hat_1 = outnet_t1(Embedding[train_t == 1])

            for i in range(args.dis_epoch):
                optimizer_D.zero_grad()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(Embedding[train_t == 0])) + torch.mean(
                    discriminator(Embedding[train_t == 1]))
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

            # loss_G
            loss_G = -torch.mean(discriminator(Embedding[train_t == 1]))

            # loss reconstruction
            loss_V = loss_fn(recon_x, Embedding, mu, log_var)
            # loss prediction
            loss_pred = F.mse_loss(y_hat_0.squeeze(), input_y0).to(device) + F.mse_loss(y_hat_1.squeeze(),
                                                                                            input_y1).to(device)
            # total loss
            loss_other = loss_G + loss_V + args.rescon * loss_pred
            optimizer_G.zero_grad()
            loss_other.backward()
            optimizer_G.step()

            if steps % args.print_steps == 0 or steps == 0:
                print(
                    "Epoches: %d, step: %d, loss_D:%.3f,loss_G:%.3f,loss_V:%.3f,loss_pre:%.3f"
                    % (epoch, steps, loss_D.detach().cpu().numpy(), loss_G.detach().cpu().numpy(),
                        loss_V.detach().cpu().numpy(), loss_pred.detach().cpu().numpy()))
                # ---------------------
                #         Test
                # ---------------------
                Reweighting.eval()
                Var_AE.eval()
                discriminator.eval()
                outnet_t0.eval()
                outnet_t1.eval()
                total_test_potential_y = torch.Tensor([]).to(device)
                total_test_potential_y_hat = torch.Tensor([]).to(device)
                test_dataset = twin_datasets(isTrain=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
                for steps, [test_x, test_t,test_potential_y] in enumerate(test_dataloader):
                    test_x = test_x.float().to(device)
                    test_t =test_t.float().to(device)
                    test_potential_y = test_potential_y.float().to(device)
                    Embedding_test= Reweighting(test_x)
                    recon_x, mu, log_var = Var_AE(Embedding_test)
                    y_hat_t0 = outnet_t0(Embedding_test)
                    y_hat_t1 = outnet_t1(Embedding_test)
                    test_potential_y_hat = torch.cat([y_hat_t0, y_hat_t1], dim=-1)
                    total_test_potential_y = torch.cat([total_test_potential_y, test_potential_y], dim=0)
                    total_test_potential_y_hat = torch.cat([total_test_potential_y_hat, test_potential_y_hat],
                                                               dim=0)

                pehe = PEHE(total_test_potential_y_hat.cpu().detach().numpy(),
                                total_test_potential_y.cpu().detach().numpy())
                ate = ATE(total_test_potential_y_hat.cpu().detach().numpy(),
                              total_test_potential_y.cpu().detach().numpy())

                print("PEHE:", pehe)
                print("ATE:", ate)
                epoch_pehe= min(pehe, epoch_pehe)
                epoch_ate = min(ate, epoch_ate)
                min_pehe = min(pehe, min_pehe)
                min_ate = min(ate, min_ate)
                Reweighting.train()
                Var_AE.train()
                discriminator.train()
                outnet_t0.train()
                outnet_t1.train()
                CBPS_list.append(epoch_pehe)
        print("PEHE:", min_pehe)
        print("ATE:", min_ate)
    train_embedding_t0 = torch.Tensor([]).to(device)
    train_embedding_t1 = torch.Tensor([]).to(device)
    train_common_t0 = torch.Tensor([]).to(device)
    train_common_t1 = torch.Tensor([]).to(device)
    for steps, [train_x, train_t, train_y, train_potential_y] in enumerate(train_dataloader):
        train_x = train_x.float().to(device)
        train_t = train_t.float().to(device)
        train_y = train_y.float().to(device)
        input_y0 = train_y[train_t == 0].float().to(device)
        input_y1 = train_y[train_t == 1].float().to(device)
        Embedding = Reweighting(train_x)
        recon_x, mu, log_var = Var_AE(Embedding)
        y_hat_0 = outnet_t0(Embedding[train_t == 0])
        y_hat_1 = outnet_t1(Embedding[train_t == 1])
        train_common_t0 = torch.cat([train_common_t0, train_x[train_t == 0]], dim=0)
        train_common_t1 = torch.cat([train_common_t1, train_x[train_t == 1]], dim=0)
        train_embedding_t0 = torch.cat([train_embedding_t0, recon_x[train_t == 0]], dim=0)
        train_embedding_t1 = torch.cat([train_embedding_t1, recon_x[train_t == 1]], dim=0)
    tsne_reduction_orgin_CBPS = TSNE(n_components=2)
    reduce_train_embedding_t0 = tsne_reduction_orgin_CBPS.fit_transform(train_embedding_t0.detach().cpu().numpy())
    reduce_train_embedding_t1 = tsne_reduction_orgin_CBPS.fit_transform(train_embedding_t1.detach().cpu().numpy())
    random_index_1 = np.random.choice(reduce_train_embedding_t0.shape[0], size=300, replace=False)
    random_index_2= np.random.choice(reduce_train_embedding_t1.shape[0], size=300, replace=False)
    train_random_t0 = reduce_train_embedding_t0[random_index_1, :]
    train_random_t1 = reduce_train_embedding_t1[random_index_2, :]
    #plt.gca().spines["bottom"].set_color("black")
    #plt.gca().spines["left"].set_color("black")
    sns.scatterplot(x=train_random_t0[:, 0], y=train_random_t0[:, 1],c='red',label="t0",)
    sns.scatterplot(x=train_random_t1[:, 0], y=train_random_t1[:, 1],c='green',label="t1")
    plt.axis("off")
    plt.legend()
    plt.savefig('CBPS_embedding.png')
    #fig, ax = plt.subplots()
    #ax.plot(epoch_list, AIPW_list,c='red',label='AIPW')
    #ax.plot(epoch_list, IPW_list,c='green',label='IPW')
    #ax.plot(epoch_list, CBPS_list,c='blue',label='CBPS')
    #ax.set_xlabel("Epoch")
    #ax.set_ylabel("PEHE")
    #plt.legend()
    #plt.savefig('Twins_Epoch_PEHE.png')
    #tsne_reduction = TSNE(n_components=2, perplexity=100, n_iter=5000)
    #train_t0 = torch.Tensor([]).to(device)
    #train_t1 = torch.Tensor([]).to(device)
    #train_t0_pre = torch.Tensor([]).to(device)
    #train_t1_pre = torch.Tensor([]).to(device)
    #画图#
    #train_t0 = tsne_reduction.fit_transform(train_x[train_t == 0].cpu().detach().numpy())
    #train_t1 = tsne_reduction.fit_transform(train_x[train_t == 1].cpu().detach().numpy())
    #train_t0_pre = tsne_reduction.fit_transform(Embedding[train_t == 0].cpu().detach().numpy())
    #train_t1_pre = tsne_reduction.fit_transform(Embedding[train_t == 1].cpu().detach().numpy())
    #plt.figure()
    #plt.scatter(train_t0,train_t0_pre, label="T0")
    #plt.scatter(train_t1,train_t1_pre, label="T1")
    #plt.savefig('AIPW.png')
def main_ihdp(args,i):
    # 构造数据集
    ihdp_dataset = ihdp_datasets(isTrain=True)
    train_dataloader = DataLoader(
        ihdp_dataset, batch_size=args.batch_size, shuffle=True)
    Reweighting = IPW(25, args.hid_dims, 25).to(device)
    Var_AE = VAE(args.input_dims, args.hid_dims, args.out_dims).to(device)
    AE= Autoencoder(args.input_dims, args.hid_dims).to(device)
    # Discriminator
    discriminator = Discriminator(args.input_dims, args.hid_dims, args.out_dims).to(device)
    outnet_t0 = OutNet(args.hid_dims, 1).to(device)
    outnet_t1 = OutNet(args.hid_dims, 1).to(device)
    optimizer_G = Adam(itertools.chain(Reweighting.parameters(), AE.parameters(),
                                       outnet_t0.parameters(), outnet_t1.parameters()),lr=args.lr)
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr)

    min_pehe = 99999
    min_ate = 99999
    epoch_list = []
    AIPW_list = []
    IPW_list = []
    CBPS_list = []
    for epoch in range(args.epoch):
        epoch_pehe = 9999
        epoch_ate = 9999
        for steps, [train_x, train_t, train_y, train_potential_y,train_mu0,train_mu1] in enumerate(train_dataloader):
            train_x = train_x[:,:,i].float().to(device).squeeze()
            train_t = train_t[:,i:i+1].float().to(device).squeeze()
            train_y = train_y[:,i:i+1].float().to(device).squeeze()
            train_potential_y = train_potential_y[:, i:i + 1].float().to(device).squeeze()
            input_y0 = train_y[train_t==0].float().to(device)
            input_y1 = train_y[train_t==1].float().to(device)
            Embedding = Reweighting(train_x)
            #recon_x, mu, log_var = Var_AE(Embedding)
            recon_x=AE(Embedding)
            y_hat_0 = outnet_t0(Embedding[train_t==0])
            y_hat_1 = outnet_t1(Embedding[train_t==1])
            for a in range(args.dis_epoch):
                optimizer_D.zero_grad()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(Embedding[train_t == 0])) + torch.mean(
                    discriminator(Embedding[train_t == 1]))
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

            # loss_G
            loss_G = -torch.mean(discriminator(Embedding[train_t == 1]))

            # loss reconstruction
            #loss_V = loss_fn(recon_x, Embedding, mu, log_var)
            loss_V=F.mse_loss(recon_x,train_x).to(device)
            # loss prediction
            loss_pred = F.mse_loss(y_hat_0.squeeze(), input_y0).to(device) + F.mse_loss(y_hat_1.squeeze(),
                                                                                        input_y1).to(device)
            # total loss
            loss_other = loss_G + loss_V + args.rescon * loss_pred
            optimizer_G.zero_grad()
            loss_other.backward()
            optimizer_G.step()

            if steps % args.print_steps == 0 or steps == 0:
                print(
                    "Epoches: %d, step: %d, loss_D:%.3f,loss_G:%.3f,loss_V:%.3f,loss_pre:%.3f"
                    % (epoch, steps, loss_D.detach().cpu().numpy(), loss_G.detach().cpu().numpy(),
                       loss_V.detach().cpu().numpy(), loss_pred.detach().cpu().numpy()))
                # ---------------------
                #         Test
                # ---------------------
                Reweighting.eval()
                AE.eval()
                discriminator.eval()
                outnet_t0.eval()
                outnet_t1.eval()
                total_test_potential_y = torch.Tensor([]).to(device)
                total_test_y = torch.Tensor([]).to(device)
                total_test_potential_y_hat = torch.Tensor([]).to(device)
                test_dataset = ihdp_datasets(isTrain=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
                for steps, [test_x, test_t, test_y,test_potential_y,test_mu0,test_mu1] in enumerate(test_dataloader):
                    test_x = test_x[:,:,i].float().to(device).squeeze()
                    test_t = test_t[:,i:i+1].float().to(device).squeeze()
                    test_y = test_y[:,i:i+1].float().to(device).squeeze()
                    test_potential_y= test_potential_y[:,i:i+1].float().to(device).squeeze()
                    test_mu0 = test_mu0[:,i:i+1].float().to(device)
                    test_mu1 = test_mu1[:,i:i+1].float().to(device)
                    Embedding_test = Reweighting(test_x)
                    #recon_x, mu, log_var = Var_AE(Embedding_test)
                    recon_x=AE(Embedding_test)
                    y_hat_t0 = outnet_t0(Embedding_test)
                    y_hat_t1 = outnet_t1(Embedding_test)
                    test_potential_y_hat = torch.cat([y_hat_t0, y_hat_t1], dim=-1)
                    total_test_potential_y_hat = torch.cat([total_test_potential_y_hat, test_potential_y_hat],dim=0)
                    total_test_potential_y= torch.cat([test_mu0,test_mu1],dim=-1)
                    total_test_y= torch.cat([total_test_y, total_test_potential_y],dim=0)
                pehe = PEHE(total_test_potential_y_hat.cpu().detach().numpy(),
                            total_test_y.cpu().detach().numpy())
                ate = ATE(total_test_potential_y_hat.cpu().detach().numpy(),
                          total_test_y.cpu().detach().numpy())
                print("PEHE:", pehe)
                print("ATE:", ate)
                epoch_pehe = min(pehe, epoch_pehe)
                epoch_ate = min(ate, epoch_ate)
                min_pehe = min(pehe, min_pehe)
                min_ate = min(ate, min_ate)
                Reweighting.train()
                AE.train()
                discriminator.train()
                outnet_t0.train()
                outnet_t1.train()
                epoch_list.append(epoch)
                AIPW_list.append(epoch_pehe)
        print("PEHE:", min_pehe)
        print("ATE:", min_ate)
    train_embedding_t0 = torch.Tensor([]).to(device)
    train_embedding_t1 = torch.Tensor([]).to(device)
    train_common_t0 = torch.Tensor([]).to(device)
    train_common_t1 = torch.Tensor([]).to(device)
    for steps, [train_x, train_t, train_y, train_potential_y, train_mu0, train_mu1] in enumerate(train_dataloader):
        train_x = train_x[:, :, i].float().to(device).squeeze()
        train_t = train_t[:, i:i + 1].float().to(device).squeeze()
        train_y = train_y[:, i:i + 1].float().to(device).squeeze()
        train_potential_y = train_potential_y[:, i:i + 1].float().to(device).squeeze()
        input_y0 = train_y[train_t == 0].float().to(device)
        input_y1 = train_y[train_t == 1].float().to(device)
        Embedding = Reweighting(train_x)
        recon_x = AE(Embedding)
        #recon_x, mu, log_var = Var_AE(Embedding)
        y_hat_0 = outnet_t0(Embedding[train_t == 0])
        y_hat_1 = outnet_t1(Embedding[train_t == 1])
        train_common_t0 = torch.cat([train_common_t0, train_x[train_t == 0]], dim=0)
        train_common_t1 = torch.cat([train_common_t1, train_x[train_t == 1]], dim=0)
        train_embedding_t0 = torch.cat([train_embedding_t0, recon_x[train_t == 0]], dim=0)
        train_embedding_t1 = torch.cat([train_embedding_t1, recon_x[train_t == 1]], dim=0)
    tsne_reduction_orgin = TSNE(n_components=2)
    reduce_train_embedding_t0 = tsne_reduction_orgin.fit_transform(train_embedding_t0.detach().cpu().numpy())
    reduce_train_embedding_t1 = tsne_reduction_orgin.fit_transform(train_embedding_t1.detach().cpu().numpy())
    # random_index_1 = np.random.choice(reduce_train_embedding_t0.shape[0], size=300, replace=False)
    # random_index_2 = np.random.choice(reduce_train_embedding_t1.shape[0], size=300, replace=False)
    # train_random_t0 = reduce_train_embedding_t0[random_index_1, :]
    # train_random_t1 = reduce_train_embedding_t1[random_index_2, :]
    # plt.gca().spines["bottom"].set_color("black")
    # plt.gca().spines["left"].set_color("black")
    sns.scatterplot(x=reduce_train_embedding_t0[:, 0], y=reduce_train_embedding_t0[:, 1], c='red', label="t0", )
    sns.scatterplot(x=reduce_train_embedding_t1[:, 0], y=reduce_train_embedding_t1[:, 1], c='green', label="t1")
    plt.axis("off")
    plt.legend()
    plt.savefig('AE_embedding.png')
def main_jobs(args,i):
    # 构造数据集
    ihdp_dataset = jobs_datasets(isTrain=True)
    train_dataloader = DataLoader(
        ihdp_dataset, batch_size=args.batch_size, shuffle=True)
    Reweighting = IPW(17, args.hid_dims, 17).to(device)
    Var_AE = VAE(args.input_dims, args.hid_dims, args.out_dims).to(device)
    AE = Autoencoder(args.input_dims, args.hid_dims).to(device)
    # Discriminator
    discriminator = Discriminator(args.input_dims, args.hid_dims, args.out_dims).to(device)
    outnet_t0 = OutNet(args.hid_dims, args.out_dims).to(device)
    outnet_t1 = OutNet(args.hid_dims, args.out_dims).to(device)
    optimizer_G = Adam(itertools.chain(Reweighting.parameters(), Var_AE.parameters(),AE.parameters(),
                                       outnet_t0.parameters(), outnet_t1.parameters()),lr=args.lr)
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr)

    min_rol = 99999
    min_att = 99999
    epoch_list = []
    AIPW_list = []
    IPW_list = []
    CBPS_list = []
    for epoch in range(args.epoch):
        epoch_pehe = 9999
        epoch_ate = 9999
        for steps, [train_x, train_t, train_y] in enumerate(train_dataloader):
            train_x = train_x[:,:,i].float().to(device).squeeze()
            train_t = train_t[:,i:i+1].float().to(device).squeeze()
            train_y = train_y[:,i:i+1].float().to(device).squeeze()
            input_y0 = train_y[train_t == 0].float().to(device)
            input_y1 = train_y[train_t == 1].float().to(device)

            Embedding = Reweighting(train_x)
            #recon_x, mu, log_var = Var_AE(Embedding)
            recon_x = AE(Embedding)
            y_hat_0 = outnet_t0(Embedding[train_t == 0])
            y_hat_1 = outnet_t1(Embedding[train_t == 1])

            for a in range(args.dis_epoch):
                optimizer_D.zero_grad()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(Embedding[train_t == 0])) + torch.mean(
                    discriminator(Embedding[train_t == 1]))
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

            # loss_G
            loss_G = -torch.mean(discriminator(Embedding[train_t == 1]))

            # loss reconstruction
            #loss_V = loss_fn(recon_x, Embedding, mu, log_var)
            loss_V = F.mse_loss(recon_x, train_x).to(device)
            # loss prediction
            loss_pred = F.mse_loss(y_hat_0.squeeze(), input_y0).to(device) + F.mse_loss(y_hat_1.squeeze(),
                                                                                        input_y1).to(device)
            # total loss
            loss_other = loss_G + loss_V + args.rescon * loss_pred
            optimizer_G.zero_grad()
            loss_other.backward()
            optimizer_G.step()

            if steps % args.print_steps == 0 or steps == 0:
                print(
                    "Epoches: %d, step: %d, loss_D:%.3f,loss_G:%.3f,loss_V:%.3f,loss_pre:%.3f"
                    % (epoch, steps, loss_D.detach().cpu().numpy(), loss_G.detach().cpu().numpy(),
                       loss_V.detach().cpu().numpy(), loss_pred.detach().cpu().numpy()))
                # ---------------------
                #         Test
                # ---------------------
                Reweighting.eval()
                AE.eval()
                discriminator.eval()
                outnet_t0.eval()
                outnet_t1.eval()
                total_test_y_0= torch.Tensor([]).to(device)
                total_test_y_hat_0= torch.Tensor([]).to(device)
                total_test_y_hat_1 = torch.Tensor([]).to(device)
                total_test_t=torch.Tensor([]).to(device)
                total_test_e=torch.Tensor([]).to(device)
                test_dataset = jobs_datasets(isTrain=False)
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
                for steps, [test_x, test_t, test_y,e] in enumerate(test_dataloader):
                    test_x = test_x[:,:,i].float().to(device).squeeze()
                    test_t = test_t[:,i:i+1].float().to(device).squeeze()
                    test_y = test_y[:,i:i+1].float().to(device).squeeze()
                    e = e[:,i:i+1].float().to(device).squeeze()
                    Embedding_test = Reweighting(test_x)
                    #recon_x, mu, log_var = Var_AE(Embedding_test)
                    recon_x = AE(Embedding)
                    y_hat_t0 = outnet_t0(Embedding_test)
                    y_hat_t1 = outnet_t1(Embedding_test)
                    test_y = test_y.unsqueeze(dim=1)
                    total_test_e = torch.cat([total_test_e, e], dim=0)
                    total_test_t =torch.cat([total_test_t, test_t], dim=0)
                    total_test_y_0= torch.cat([total_test_y_0, test_y], dim=0)
                    total_test_y_hat_0= torch.cat([total_test_y_hat_0,y_hat_t0 ], dim=0)
                    total_test_y_hat_1 = torch.cat([total_test_y_hat_1, y_hat_t1], dim=0)
                #att= ATT(total_test_y_0.cpu().detach().numpy(),total_test_y_hat_0.cpu().detach().numpy(),total_test_y_hat_1.cpu().detach().numpy(),total_test_t.cpu().detach().numpy(),total_test_e.cpu().detach().numpy())
                #rol= ROL(total_test_y_0.cpu().detach().numpy(), total_test_y_hat_0.cpu().detach().numpy(),total_test_y_hat_1.cpu().detach().numpy(), total_test_t.cpu().detach().numpy(),total_test_e.cpu().detach().numpy())
                att,rol=Jobs_metric(total_test_y_0.cpu().detach().numpy(),total_test_y_hat_0.cpu().detach().numpy(),total_test_y_hat_1.cpu().detach().numpy(),total_test_t.cpu().detach().numpy(),total_test_e.cpu().detach().numpy())
                print("ATT:", att)
                print("RoL:",rol)
                min_att = min(att, min_att)
                min_rol = min(rol, min_rol)
                Reweighting.train()
                AE.train()
                discriminator.train()
                outnet_t0.train()
                outnet_t1.train()
                epoch_list.append(epoch)
                AIPW_list.append(epoch_pehe)
        print("ATT:", min_att)
        print("Rol:", min_rol)
    train_embedding_t0 = torch.Tensor([]).to(device)
    train_embedding_t1 = torch.Tensor([]).to(device)
    train_common_t0 = torch.Tensor([]).to(device)
    train_common_t1 = torch.Tensor([]).to(device)
    for steps, [train_x, train_t, train_y] in enumerate(train_dataloader):
        train_x = train_x[:, :, i].float().to(device).squeeze()
        train_t = train_t[:, i:i + 1].float().to(device).squeeze()
        train_y = train_y[:, i:i + 1].float().to(device).squeeze()
        input_y0 = train_y[train_t == 0].float().to(device)
        input_y1 = train_y[train_t == 1].float().to(device)
        Embedding = Reweighting(train_x)
        #recon_x, mu, log_var = Var_AE(Embedding)
        recon_x = AE(Embedding)
        y_hat_0 = outnet_t0(Embedding[train_t == 0])
        y_hat_1 = outnet_t1(Embedding[train_t == 1])
        train_common_t0 = torch.cat([train_common_t0, train_x[train_t == 0]], dim=0)
        train_common_t1 = torch.cat([train_common_t1, train_x[train_t == 1]], dim=0)
        train_embedding_t0 = torch.cat([train_embedding_t0, recon_x[train_t == 0]], dim=0)
        train_embedding_t1 = torch.cat([train_embedding_t1, recon_x[train_t == 1]], dim=0)
    tsne_reduction_orgin = TSNE(n_components=2)
    reduce_train_embedding_t0 = tsne_reduction_orgin.fit_transform(train_embedding_t0.detach().cpu().numpy())
    reduce_train_embedding_t1 = tsne_reduction_orgin.fit_transform(train_embedding_t1.detach().cpu().numpy())
    # reduce_train_common_t0 = tsne_reduction_orgin.fit_transform(train_common_t0.detach().cpu().numpy())
    # reduce_train_common_t1 = tsne_reduction_orgin.fit_transform(train_common_t1.detach().cpu().numpy())
    # random_index_1 = np.random.choice(reduce_train_embedding_t0.shape[0], size=300, replace=False)
    # random_index_2 = np.random.choice(reduce_train_embedding_t1.shape[0], size=300, replace=False)
    # train_random_t0 = reduce_train_embedding_t0[random_index_1, :]
    # train_random_t1 = reduce_train_embedding_t1[random_index_2, :]
    # plt.gca().spines["bottom"].set_color("black")
    # plt.gca().spines["left"].set_color("black")
    sns.scatterplot(x=reduce_train_embedding_t0[:, 0], y=reduce_train_embedding_t0[:, 1], c='red', label="t0", )
    sns.scatterplot(x=reduce_train_embedding_t1[:, 0], y=reduce_train_embedding_t1[:, 1], c='green', label="t1")
    # sns.scatterplot(x=reduce_train_common_t0[:, 0], y=reduce_train_common_t0[:, 1], c='red', label="t0", )
    # sns.scatterplot(x=reduce_train_common_t1[:, 0], y=reduce_train_common_t1[:, 1], c='green', label="t1")
    plt.axis("off")
    plt.legend()
    plt.savefig('AE_embedding.png')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--input_dims", default=17, type=int)
    parser.add_argument("--hid_dims", default=17, type=int)
    parser.add_argument("--out_dims", default=1, type=int)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--print_steps", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=int)
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--rescon", type=int, default=10, help="weights of rescon loss")
    parser.add_argument("--dis_epoch", type=int, default=10, help="discrimator epoch")

    args = parser.parse_args()
    print(args)

    if (torch.cuda.is_available()):
        print("GPU is ready \n")
    else:
        print("CPU is ready \n")
    #main_ihdp(args,0)
    main_jobs(args,0)
    # main(args)
    #mainSubITE(args)