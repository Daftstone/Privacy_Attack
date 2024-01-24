import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.nn import GCNConv as GCNConv
import scipy.sparse as sp
import numpy as np
import tqdm
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from parse import FLAGS


class MyDataset(torch_geometric.data.Dataset):
    def __init__(self, dataset, name='train'):
        super().__init__()

        self.dataset = dataset
        adj = dataset.adj.tocoo()
        rows = adj.row
        cols = adj.col

        self.edge_index = torch.tensor(np.concatenate([rows[None, :], cols[None, :]], axis=0)).long()
        self.x = torch.tensor(dataset.feature)

        if (name == 'train'):
            self.all_data = self.dataset.all_train_data
            self.all_label = self.dataset.all_train_label
        elif (name == 'val'):
            self.all_data = self.dataset.all_val_data
            self.all_label = self.dataset.all_val_label
        else:
            self.all_data = self.dataset.all_test_data
            self.all_label = self.dataset.all_test_label
        self.all_data = torch.tensor(self.all_data)
        self.all_label = torch.tensor(self.all_label[:, None]).float()
        print(len(self.all_data))

    def len(self):
        return len(self.all_data)

    def get(self, idx):
        return self.all_data[idx], self.all_label[idx]


class GCN(nn.Module):
    def __init__(self, dataset):

        super(GCN, self).__init__()

        self.device = torch.device("cuda", int(FLAGS.gpu))

        self.dataset = dataset
        pre_layer = self.dataset.feature.shape[1]

        self.conv1 = torch.nn.ModuleList()
        for layer in [64]:
            self.conv1.append(GCNConv(pre_layer, layer))
            pre_layer = layer

        self.linear = nn.Linear(pre_layer * 2, 16)
        self.linear1 = nn.Linear(16, 1)

        self.train_dataloader = MyDataset(dataset)
        self.val_dataloader = MyDataset(dataset, 'val')
        self.test_dataloader = MyDataset(dataset, 'test')

    def reset_parameters(self):
        self.linear.reset_parameters()
        for lin in self.conv1:
            lin.reset_parameters()

    def forward(self, x, edge_index, index):
        node_embed = x
        for conv in self.conv1:
            node_embed = conv(node_embed, edge_index)
            node_embed = F.dropout(node_embed, 0.5, self.training)
            node_embed = F.leaky_relu(node_embed)

        x_i = torch.index_select(node_embed, 0, index[:, 0])
        x_j = torch.index_select(node_embed, 0, index[:, 1])

        # out = F.sigmoid(torch.sum(x_i * x_j, dim=1, keepdim=True))

        out = torch.concat([x_i, x_j], dim=1)
        out = F.leaky_relu(self.linear(out))
        out = self.linear1(out)
        out = F.sigmoid(out)

        return out

    def my_train(self, epochs, batch_size):
        self.to(self.device)
        edge_index = self.train_dataloader.edge_index.to(self.device)
        features = self.train_dataloader.x.to(self.device).float()
        # self.features = (features - torch.mean(features, dim=0, keepdim=True)) / (
        #         torch.std(features, dim=0, keepdim=True) + 1e-8)
        self.features = features

        criterion = torch.nn.MSELoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005, weight_decay=0.)  # Define optimizer.

        val_list = []
        test_list = []
        val_loss_list = []
        test_loss_list = []

        for epoch in tqdm.tqdm(range(1, epochs + 1), 'epochs'):
            pbar = tqdm.tqdm(
                torch_geometric.loader.dataloader.DataLoader(self.train_dataloader, batch_size=batch_size,
                                                             num_workers=0))
            count = 0
            for x, y in pbar:
                self.train()
                count += 1
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.forward(self.features, edge_index, x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                if (count % 50 == 1):
                    pred = np.mean((out.detach().cpu().numpy() > 0.5) == y.cpu().numpy())
                    pbar.set_postfix(loss=loss.detach().cpu().item(), acc=pred)

            self.eval()
            train_loss, train_acc = self.evaluate(self.train_dataloader)
            val_loss, val_acc = self.evaluate(self.val_dataloader)
            test_loss, test_acc = self.evaluate(self.test_dataloader)
            print(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

            val_list.append(val_acc)
            test_list.append(test_acc)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)

        idx = np.argmax(np.array(val_list))
        idx1 = np.argmax(np.array(val_loss_list))
        try:
            cur_results = np.load("results/acc_%s_%s_%s_%s_%d_%d_%d_%.2f.npy" % (
                FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                FLAGS.ratio))
            cur_results = list(cur_results)
            cur_results1 = np.load("results/loss_%s_%s_%s_%s_%d_%d_%d_%.2f.npy" % (
                FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                FLAGS.ratio))
            cur_results1 = list(cur_results1)
        except:
            cur_results = []
            cur_results1 = []
        cur_results.append(test_list[idx])
        cur_results1.append(test_loss_list[idx1])
        np.save("results/acc_%s_%s_%s_%s_%d_%d_%d_%.2f.npy" % (
                FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                FLAGS.ratio),
                cur_results)
        np.save("results/loss_%s_%s_%s_%s_%d_%d_%d_%.2f.npy" % (
                FLAGS.defense, FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt,
                FLAGS.ratio),
                cur_results1)
        print('best', test_list[idx])

    def evaluate(self, dataloader, batch_size=2048):
        edge_index = dataloader.edge_index.to(self.device)
        # features = dataloader.x.to(self.device).float()
        criterion = torch.nn.MSELoss()  # Define loss criterion.
        loss_list = []
        acc_list = []
        for x, y in iter(
                torch_geometric.loader.dataloader.DataLoader(dataloader, batch_size=batch_size, num_workers=1)):
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.forward(self.features, edge_index, x)
            loss = criterion(out, y)
            loss_list.append(loss.detach().cpu().numpy())
            acc_list.append(np.mean((out.detach().cpu().numpy() > 0.5) == y.cpu().numpy()))
        return np.mean(loss_list), np.mean(acc_list)
