import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.data import Data


class GCN3Layer(nn.Module):

    def __init__(self, num_features):
        super(GCN3Layer, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.linear = Linear(16, 1)
        self.relu = ReLU()
        self.dropout = Dropout(0.2)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x_conv1 = self.dropout(self.relu(self.conv1(x, edge_index)))
        x_conv2 = self.dropout(self.relu(self.conv2(x_conv1, edge_index)))
        x_conv3 = self.dropout(self.relu(self.conv3(x_conv2, edge_index)))
        out = torch.sigmoid(self.linear(x_conv3))

        return out


class ResGCN3Layer(nn.Module):

    def __init__(self, num_features):
        super(ResGCN3Layer, self).__init__()

        self.conv1 = GCNConv(num_features, 64)

        self.skip0_2 = Linear(num_features, 32)
        self.conv2 = GCNConv(64, 32)

        self.skip1_3 = Linear(64, 16)
        self.conv3 = GCNConv(32, 16)

        self.linear = Linear(16, 1)

        self.relu = ReLU()
        self.dropout = Dropout(0.2)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x_conv1 = self.dropout(self.relu(self.conv1(x, edge_index)))
        x_conv2 = self.dropout(self.relu(self.conv2(x_conv1, edge_index) + self.skip0_2(x)))
        x_conv3 = self.dropout(self.relu(self.conv3(x_conv2, edge_index) + self.skip1_3(x_conv1)))
        out = torch.sigmoid(self.linear(x_conv3))

        return out


class DenseGCN3Layer(nn.Module):

    def __init__(self, num_features):
        super(DenseGCN3Layer, self).__init__()

        self.conv1 = GCNConv(num_features, 64)

        self.skip0_2 = Linear(num_features, 32)
        self.conv2 = GCNConv(64, 32)

        self.skip0_3 = Linear(num_features, 16)
        self.skip1_3 = Linear(64, 16)
        self.conv3 = GCNConv(32, 16)

        self.linear = Linear(16, 1)

        self.relu = ReLU()
        self.dropout = Dropout(0.2)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x_conv1 = self.dropout(self.relu(self.conv1(x, edge_index)))
        x_conv2 = self.dropout(self.relu(self.conv2(x_conv1, edge_index) + self.skip0_2(x)))
        x_conv3 = self.dropout(self.relu(self.conv3(x_conv2, edge_index) + self.skip0_3(x) + self.skip1_3(x_conv1)))
        out = torch.sigmoid(self.linear(x_conv3))

        return out
