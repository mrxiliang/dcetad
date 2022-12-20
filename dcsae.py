from torch import nn
from dim_net import DimNet

# initialize
epochs = 0
en_input1 = 0
en_output1 = 0
en_input2 = 0
en_output2 = 0
de_input1 = 0
de_output1 = 0
de_input2 = 0
de_output2 = 0


class DCSparseAutoEncoder(nn.Module):
    def __init__(self):
        super(DCSparseAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(en_input1, en_output1),
            nn.LeakyReLU(),
            nn.Linear(en_input2, en_output2),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(de_input1, de_output1),
            nn.LeakyReLU(),
            nn.Linear(de_input2, de_output2),
            nn.Tanh()
        )
        self.eca = DimNet(kernel_size=15)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.unsqueeze(axis=2)
        encoded = encoded.unsqueeze(axis=3)
        encoded = self.eca(encoded)
        encoded = encoded.squeeze(axis=3)
        encoded = encoded.squeeze(axis=2)
        decoded = self.decoder(encoded)
        return encoded, decoded
