import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.fullCon1 = nn.Linear(in_features=117, out_features=50)

        self.fullCon2 = nn.Linear(in_features=50, out_features=1)

    #         self.fullCon3 = nn.Linear(in_features=50, out_features=10)

    #         self.fullCon4 = nn.Linear(in_features=10, out_features=1)

    #         self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.tanh(self.fullCon1(x))
        #         x = F.dropout(x)
        x = F.sigmoid(self.fullCon2(x))
        #         x = F.dropout(x)
        #         x = F.sigmoid(self.fullCon3(x))
        #         x = F.sigmoid(self.fullCon4(x))
        return x
