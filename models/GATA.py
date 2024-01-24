import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.nn import GATv2Conv as GCNConv
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
            self.all_data = self.dataset.train_idx
        elif (name == 'val'):
            self.all_data = self.dataset.val_idx
        else:
            self.all_data = self.dataset.test_idx
        self.all_data = torch.tensor(self.all_data)
        print(len(self.all_data))

    def len(self):
        return len(self.all_data)

    def get(self, idx):
        return self.all_data[idx]


class GCN(nn.Module):
    def __init__(self, dataset):

        super(GCN, self).__init__()

        self.device = torch.device("cuda", int(FLAGS.gpu))

        self.dataset = dataset
        pre_layer = self.dataset.feature.shape[1]

        self.conv1 = torch.nn.ModuleList()
        for layer in [128, 32]:
            self.conv1.append(GCNConv(pre_layer, layer))
            pre_layer = layer

        self.linear = nn.Linear(pre_layer, 2)

        self.train_dataloader = MyDataset(dataset)
        self.val_dataloader = MyDataset(dataset, 'val')
        self.test_dataloader = MyDataset(dataset, 'test')

    def reset_parameters(self):
        self.linear.reset_parameters()
        for lin in self.conv1:
            lin.reset_parameters()

    def forward(self, x, edge_index):
        node_embed = x

        for conv in self.conv1:
            node_embed = conv(node_embed, edge_index)
            node_embed = F.leaky_relu(node_embed)

        out = self.linear(node_embed)
        out = F.softmax(out)

        return out

    def my_train(self, epochs, batch_size):
        # epochs = 500
        batch_size = 256
        self.to(self.device)
        edge_index = self.train_dataloader.edge_index.to(self.device)
        features = self.train_dataloader.x.to(self.device).float()

        criterion = torch.nn.BCELoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=5e-4)  # Define optimizer.

        val_list = []
        test_list = []

        secret_label = torch.tensor(self.dataset.secret_label).to(self.device).float()

        for epoch in tqdm.tqdm(range(1, epochs + 1), 'epochs'):
            pbar = tqdm.tqdm(
                torch_geometric.loader.dataloader.DataLoader(self.train_dataloader, batch_size=batch_size,
                                                             num_workers=0))
            count = 0
            for x in pbar:
                self.train()
                count += 1
                x = x.to(self.device)
                out = self.forward(features, edge_index)
                out = torch.index_select(out, 0, x)
                y = torch.index_select(secret_label, 0, x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                if (count % 10 == 0):
                    pred = np.mean(np.argmax(out.detach().cpu().numpy(), axis=1) == np.argmax(y.cpu().numpy(), axis=1))
                    pbar.set_postfix(loss=loss.detach().cpu().item(), acc=pred)

            self.eval()
            train_loss, train_acc = self.evaluate(edge_index, features, self.dataset.train_idx,
                                                  secret_label[self.dataset.train_idx])
            val_loss, val_acc = self.evaluate(edge_index, features, self.dataset.val_idx,
                                              secret_label[self.dataset.val_idx])
            test_loss, test_acc = self.evaluate(edge_index, features, self.dataset.test_idx,
                                                secret_label[self.dataset.test_idx])
            print(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

            val_list.append(val_acc)
            test_list.append(test_acc)

            # if (epoch - np.argmax(val_list) > 5):
            #     break

        idx = np.argmax(np.array(val_list))
        # try:
        #     cur_results = np.load("results/%s_%s_%s_%d_%d_%d_%.2f.npy" % (
        #         FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio))
        #     cur_results = list(cur_results)
        # except:
        #     cur_results = []
        # cur_results.append(test_list[idx])
        print('best', test_list[idx])
        np.save("results/%s_%s_%s_%d_%d_%d_%.2f.npy" % (
            FLAGS.model, FLAGS.surrogate, FLAGS.dataset, FLAGS.num, FLAGS.encrypt, FLAGS.decrypt, FLAGS.ratio),
                test_list[idx])

    def evaluate(self, edge_index, features, idx, y):
        criterion = torch.nn.BCELoss()  # Define loss criterion.
        out = self.forward(features, edge_index)[idx]
        loss = criterion(out, y).detach().cpu().numpy()
        acc = np.mean(np.argmax(out.detach().cpu().numpy(), axis=1) == np.argmax(y.cpu().numpy(), axis=1))
        return loss, acc
