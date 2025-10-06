import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn import metrics
from models import DenseGCN3Layer
from preprocessing import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    time_steps = load_dataset('data/elliptic_bitcoin_dataset')
    for step in time_steps:
        time_steps[step] = time_steps[step].to(device)

    train = list(time_steps.values())[:34]
    test = list(time_steps.values())[34:]

    num_features = time_steps[1].x.shape[1]
    model = DenseGCN3Layer(num_features).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model.train()
    for epoch in range(275):
        epoch_loss = 0
        for step in train:
            optimizer.zero_grad()
            out = model(step)
            loss = F.binary_cross_entropy(out[step.mask], step.y[step.mask])
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        if epoch % 10 == 9:
            print('Epoch {}: loss {}'.format(epoch + 1, epoch_loss / len(train)))

    model.eval()
    true = torch.Tensor([]).to(device)
    predicted = torch.Tensor([]).to(device)
    for step in test:
        pred = model(step).round()
        predicted = torch.cat((predicted, pred[step.mask]), dim=0)
        true = torch.cat((true, step.y[step.mask]), dim=0)
    predicted = predicted.detach().cpu()
    true = true.cpu()
    print('Test accuracy: {:.4f}'.format(metrics.accuracy_score(true, predicted)))
    print('Test precision: {:.4f}'.format(metrics.precision_score(true, predicted)))
    print('Test recall: {:.4f}'.format(metrics.recall_score(true, predicted)))
    print('Test F1: {:.4f}'.format(metrics.f1_score(true, predicted)))
    print('Test micro-average F1: {:.4f}'.format(metrics.f1_score(true, predicted, average='micro')))
