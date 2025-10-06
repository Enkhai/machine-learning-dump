from torch import nn
import torch
import torch.nn.functional as F


# an example of a classifier for the MNIST dataset
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # linear layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # we apply dropout with 20% probability in each layer to increase model generalization capability
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        x = F.dropout(F.relu(self.fc2(x)), p=0.2)
        x = F.dropout(F.relu(self.fc3(x)), p=0.2)
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


# another classifier example using the Sequential module
model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(64, 10),
                      nn.LogSoftmax())

# the model state can be returned in a dictionary
print(model.state_dict())
# which we can save
torch.save(model.state_dict(), 'checkpoint.pth')
# and load
state_dict = torch.load('checkpoint.pth')
# from which we can load back to a model
model.load_state_dict(state_dict)
# be careful however, your model dimensions need to match those of the state dict

# another way
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = a_random_Classifier(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    return model

model = load_checkpoint('checkpoint.pth')
print(model)