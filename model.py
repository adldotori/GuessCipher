import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset): 
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx]).cuda()
        y = torch.FloatTensor(self.y_data[idx]).cuda()
        return x, y

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

        self.model = [Block(156, 300)]
        for i in range(5):
            self.model.append(Block(300, 300))
        self.model = nn.Sequential(*self.model)
        self.fc = nn.Sequential(
            nn.Linear(300, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    
    x_data = []
    y_data = []

    f_c = open('.\\dataset\\ciphertext.txt', 'r')
    f_p = open('.\\dataset\\plaintext.txt', 'r')
    f_k = open('.\\dataset\\key.txt', 'r')

    for i in range(10000):
        tmp_x = eval(f_p.readline()) + eval(f_c.readline()) + eval(f_k.readline())[28:]
        tmp_y = [1]
        x_data.append(tmp_x)
        y_data.append(tmp_y)
    for i in range(10000):
        tmp_x = [random.randint(0,1) for _ in range(156)]
        tmp_y = [0]
        x_data.append(tmp_x)
        y_data.append(tmp_y)

    model = Model()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02) 

    dataset = CustomDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size = 64, shuffle = True)
    nb_epochs = 10000
    for epoch in range(nb_epochs + 1):
        for batch_idx, samples in enumerate(dataloader):
            # print(batch_idx)
            x_train, y_train = samples
            # H(x) 계산
            prediction = model(x_train)
            # cost 계산
            cost = F.l1_loss(prediction, y_train)

            # cost로 H(x) 계산
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
            ))