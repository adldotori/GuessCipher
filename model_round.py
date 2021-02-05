import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from DES import make_train_data_IP, make_train_data_round, make_train_data_sbox

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
    known_key_bits = 56
    batchsize = 32
    nb_epochs = 1

    path_A = 'des_round_A.pt'
    path_B = 'des_round_B.pt'
    model_A = Model(64, 512, known_key_bits, 4)  # **********************
    model_B = Model(512, 64, 0, 3)
    
    model_A.to(device)
    model_B.to(device)

    optimizer_A = torch.optim.Adam(model_A.parameters(), lr=0.000001) 
    optimizer_B = torch.optim.Adam(model_B.parameters(), lr=0.00002) 
    
    count_A = 0

    while(True):
        
        x_data = []
        y_data = []
        model_A.eval()
        for count_B in range(20):
            for _ in range(2048):
                tmp_plain, tmp_cipher, tmp_key = make_train_data_round() # **********************
                x_data.append(tmp_plain + tmp_key[:known_key_bits])
                y_data.append(tmp_cipher)
            dataset = CustomDataset(x_data, y_data)
            dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

            for epoch in range(nb_epochs):
                for batch_idx, samples in enumerate(dataloader):
                    # print(batch_idx)
                    x_train, y_train = samples
                    # H(x) 계산
                    mid_train = model_A(x_train)
                    prediction = model_B(mid_train)
                    # cost 계산
                    cost = F.l1_loss(prediction, y_train)

                    # cost로 H(x) 계산
                    optimizer_B.zero_grad()
                    cost.backward()
                    optimizer_B.step()

                    if batch_idx == 0 and epoch == 0:
                        torch.save(model_A.state_dict(), path_A)
                        torch.save(model_B.state_dict(), path_B)
                        print('count_A: {:2d}  count_B: {:2d}  Cost: {:.6f}'.format(count_A, count_B, cost.item()))

                    if cost.item() < 0.05:
                        torch.save(model_A.state_dict(), path_A)
                        torch.save(model_B.state_dict(), path_B)
                        exit()

        mid_train = model_A(x_train)
        prediction = model_B(mid_train)
        cost = F.l1_loss(prediction, y_train)

        model_A.train()

        print(cost.item())

        optimizer_A.zero_grad()
        cost.backward()
        optimizer_A.step()
        
        count_A += 1
