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
    def __init__(self, known_key_bits):
        super().__init__()

        self.model = [Block(64+known_key_bits, 512)]
        for i in range(3):
            self.model.append(Block(512, 512))
        self.model = nn.Sequential(*self.model)
        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    
    x_data = []
    y_data = []
    test_x_data = []
    test_y_data = []

    known_key_bits = 56
    data_start = 10000

    f_c = open('.\\dataset\\ciphertext.txt', 'r')
    f_p = open('.\\dataset\\plaintext.txt', 'r')
    f_k = open('.\\dataset\\key.txt', 'r')

    for _ in range(data_start):
        f_p.readline()
        f_c.readline()
        f_k.readline()

    for i in range(100000):
        tmp_x = eval(f_p.readline()) + eval(f_k.readline())[:known_key_bits]
        tmp_y = eval(f_c.readline())
        x_data.append(tmp_x)
        y_data.append(tmp_y)
        #y_data.append([10*i for i in tmp_y])

        tmp_x = eval(f_p.readline()) + eval(f_k.readline())[:known_key_bits]
        tmp_y = eval(f_c.readline())
        test_x_data.append(tmp_x)
        test_y_data.append(tmp_y)
        #test_y_data.append([10*i for i in tmp_y])
        

    model = Model(known_key_bits)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00002) 

    dataset = CustomDataset(x_data, y_data)
    test_dataset = CustomDataset(test_x_data, test_y_data)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 25)
    nb_epochs = 5
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
            if cost.item() < 0.05 and epoch > 0:
                break
            if batch_idx % 500 == 0:
                print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                    epoch, nb_epochs, batch_idx+1, len(dataloader),
                    cost.item()
                    ))
        if cost.item() < 0.05 and epoch > 0:
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, batch_idx+1, len(dataloader),
                cost.item()
                ))
            break

    model.eval()
    for batch_idx, samples in enumerate(test_dataloader):
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.l1_loss(prediction, y_train)
        print('Cost: {:.6f}'.format(cost.item()))
        if batch_idx == 100:
            break
