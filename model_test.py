import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from DES import make_train_data_IP, make_train_data_round, make_train_data_sbox, make_train_data_simple

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
    def __init__(self, in_channel, out_channel, known_key_bits, hidden_layers):
        super().__init__()

        self.model = [Block(in_channel + known_key_bits, 512)]
        for i in range(hidden_layers):
            self.model.append(Block(512, 512))
        self.model = nn.Sequential(*self.model)
        self.fc = nn.Sequential(
            nn.Linear(512, out_channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    known_key_bits = 0
    batchsize = 32
    rounds = 4
    nb_epochs = 2

    model_A = Model(48, 32, known_key_bits, 5)  # **********************
    
    model_A.to(device)

    optimizer_A = torch.optim.Adam(model_A.parameters(), lr=0.000003) 
    
    count_A = 0

    while(True):
        model_A.eval()
        for count_B in range(100):
            x_data = []
            y_data = []
            for _ in range(1024):
                tmp_plain, tmp_cipher = make_train_data_simple() # **********************
                x_data.append(tmp_plain)
                y_data.append(tmp_cipher)
            dataset = CustomDataset(x_data, y_data)
            dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

            for epoch in range(nb_epochs):
                for batch_idx, samples in enumerate(dataloader):
                    # print(batch_idx)
                    x_train, y_train = samples
                    # H(x) 계산
                    prediction = model_A(x_train)
                    # cost 계산
                    cost = F.l1_loss(prediction, y_train)

                    # cost로 H(x) 계산
                    optimizer_A.zero_grad()
                    cost.backward()
                    optimizer_A.step()

                    if batch_idx == 0 and epoch == 0:
                        print('count_A: {:2d}  count_B: {:2d}  Cost: {:.6f}'.format(count_A, count_B, cost.item()))

        count_A += 1
