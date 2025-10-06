import torch
from torch_geometric.data import Data
import pandas as pd


def load_dataset(folder):
    df_classes = pd.read_csv(folder + '/elliptic_txs_classes.csv')
    df_features = pd.read_csv(folder + '/elliptic_txs_features.csv')
    df_edgelist = pd.read_csv(folder + '/elliptic_txs_edgelist.csv')

    df_classes.columns = ['ID', 'Class']
    # 0 for licit, 1 for illicit, 2 for unknown
    df_classes['Class'] = df_classes['Class'].map({'2': 0, '1': 1, 'unknown': 2})
    transaction_cols = [f'Trans_feature_{i}' for i in range(93)]
    graph_cols = [f'Graph_feature_{i}' for i in range(72)]
    df_features.columns = ['ID', 'Time step'] + transaction_cols + graph_cols
    df_features.drop(columns=graph_cols, inplace=True)
    df_edgelist.columns = ['ID1', 'ID2']

    df_class_and_features = pd.merge(df_classes, df_features, on='ID')
    df_class_and_features.set_index('ID', inplace=True)

    time_steps = {}
    for step in df_class_and_features['Time step'].unique():
        print('Processing step {}...'.format(step))
        graph_features = df_class_and_features[df_class_and_features['Time step'] == step].drop(columns='Time step')
        graph_edges = df_edgelist[df_edgelist['ID1'].isin(graph_features.index) &
                                  df_edgelist['ID2'].isin(graph_features.index)]

        old_id = graph_features.index
        graph_features.reset_index(drop=True, inplace=True)
        new_id = graph_features.index
        old2new = dict(zip(old_id, new_id))
        graph_edges = graph_edges.replace(old2new)

        x = torch.from_numpy(graph_features.drop(columns='Class').to_numpy()).float()
        edges = torch.from_numpy(graph_edges.to_numpy().T)
        y = torch.from_numpy(graph_features['Class'].to_numpy().reshape((-1, 1))).float()
        mask = y.flatten() != 2
        time_steps[step] = Data(x=x, edge_index=edges, y=y, mask=mask)

    return time_steps
