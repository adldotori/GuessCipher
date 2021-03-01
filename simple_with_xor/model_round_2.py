import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from DES import make_train_data_simple2

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

        self.model = [Block(in_channel + known_key_bits, 256)]
        for i in range(hidden_layers):
            self.model.append(Block(256, 256))
        self.model = nn.Sequential(*self.model)
        self.fc = nn.Sequential(
            nn.Linear(256, out_channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    plain_len = 48
    cipher_len = 32
    known_key_bits = 48

    batchsize = 32
    nb_epochs = 2
    model_num = 7

    models = []
    paths = []
    optimizers = []

    for i in range(model_num):
        paths.append('.\model\des_simple2_{}.pt'.format(i))

    models.append(Model(plain_len, plain_len + known_key_bits, known_key_bits, 2))
    for i in range(model_num - 2):
        models.append(Model(plain_len + known_key_bits, plain_len + known_key_bits, plain_len + known_key_bits, 2))
    models.append(Model(plain_len + known_key_bits, cipher_len, plain_len + known_key_bits, 2))

    for i in range(model_num):
        models[i].load_state_dict(torch.load(paths[i]))
        
    for model in models:
        model.to(device)
        optimizers.append(torch.optim.Adam(model.parameters(), lr=0.000007) )
    
    for optimizer in optimizers:
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998) 

    count = 0
    while(True):
        for model in models:
            model.train()
        
        for i in range(model_num):
            for tmp_count in range(200):
                x_data = []
                y_data = []
                for _ in range(1024):
                    tmp_plain, tmp_cipher, tmp_key = make_train_data_simple2() # **********************
                    x_data.append(tmp_plain + tmp_key[:known_key_bits])
                    y_data.append(tmp_cipher)
                dataset = CustomDataset(x_data, y_data)
                dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
                for epoch in range(nb_epochs):
                    for batch_idx, samples in enumerate(dataloader):
                        x_train, y_train = samples
                        # H(x) 계산
                        prediction = x_train
                        for model in models[:-1]:
                            prediction = torch.cat([model(prediction), x_train], dim = 1)
                        prediction = models[-1](prediction)
                        # cost 계산
                        cost = F.l1_loss(prediction, y_train)

                        # cost로 H(x) 계산
                        for optimizer in optimizers[i:]:
                            optimizer.zero_grad()
                        cost.backward()
                        for optimizer in optimizers[i:]:
                            optimizer.step()

                        if batch_idx == 0 and epoch == 0:
                            for j in range(model_num - i):
                                torch.save(models[i+j].state_dict(), paths[i+j])
                            print('count : {:2d}  count_{}: {:2d}  Cost: {:.6f}'.format(count, i, tmp_count, cost.item()))
        count += 1