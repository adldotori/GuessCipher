import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(out_channels)

        self.downsample = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        out = self.relu1(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        # print(out.shape, x.shape)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return self.relu2(out)
         
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = [Block(156, 128)]
        for i in range(5):
            self.model.append(Block(128, 128))
        self.model = nn.Sequential(*self.model)
        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    batch_size = 64

    model = Model()
    model.to(device)
    sample = torch.randint(0,2, (batch_size, 156,)).to(device).type(torch.FloatTensor)
    res = model(sample)
    print(res.shape)
    print(res)