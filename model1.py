import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from DES import make_train_data_IP, make_train_data_round

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
    def __init__(self, known_key_bits, hidden_layers):
        super().__init__()

        self.model = [Block(64+known_key_bits, 512)]
        for i in range(hidden_layers):
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
    known_key_bits = 0
    batchsize = 32
    rounds = 4
    nb_epochs = 3

    #ip
    ip_hidden_layers = 1
    path_ip = "des_ip.pt"

    #round
    round_hidden_layers = 3
    path_round = "des_round.pt"

    model = Model(known_key_bits, round_hidden_layers)  # **********************
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    count = 0


    while(True):
        x_data = []
        y_data = []

        for _ in range(1000):
            tmp_plain, tmp_cipher = make_train_data_round() # **********************
            x_data.append(tmp_plain)
            y_data.append(tmp_cipher)

        dataset = CustomDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
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

                if batch_idx == 0 and epoch == 0:
                    print('Epoch {:4d}  Cost: {:.6f}'.format(
                        count,
                        cost.item()
                        ))
                if cost.item() < 0.002:
                    torch.save(model.state_dict(), path_round)      # **********************
                    exit()
        if count == 10:
            break
        count += 1

    model.eval()
    for _ in range(1000):
        tmp_plain, tmp_cipher = make_train_data_round() # **********************
        x_data.append(tmp_plain)
        y_data.append(tmp_cipher)
        
    dataset = CustomDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        prediction = model(x_train)
        prediction = torch.round(prediction)
        cost = F.l1_loss(prediction, y_train)
        prediction_list = prediction.tolist()
        y_list = y_train.tolist()
        print('Cost: {:.6f}'.format(cost.item()))
        for i in range(len(prediction_list)):
            cnt = 0
            for j in range(len(prediction_list[i])):
                if prediction_list[i][j] == y_list[i][j]:
                    cnt += 1
            print('accurate bit : {:2d} all bit : {:2d}'.format(cnt, len(prediction_list[i])))


