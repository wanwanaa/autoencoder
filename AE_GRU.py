import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 64
TIME_STEP = 28 # run time step
INPUT_SIZE = 28
HIDDEN_SIZE = 64
OUTPUT_SIZE = 28
LR = 0.01

train_data = dsets.MNIST(root='/mnist', train=True, transform=transforms.ToTensor(), download=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = dsets.MNIST(root='/mnist', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:10]/255


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # encoder
        self.f1 = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, 1, batch_first=True)

        # decoder
        self.f2 = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, 1, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.ReLU()
        )

    def encoder(self, x):
        code, _ = self.f1(x)
        return code[:, -1, :].view(1, -1, HIDDEN_SIZE)

    # def decoder(self, x, h):
    #     out, h = self.f2(x, h)
    #     return F.relu(out), h

    def decoder(self, x, h):
        # print('x:', x.size())
        # print('h:', h.size())
        out, h = self.f2(x, h)
        # result = []
        # for i in range(out.size(1)):
        #     result.append(self.output(out[:, i, :]))
        # out = torch.stack(result, dim=0)
        return out, h

    def output_layer(self, x):
        result = []
        for i in range(x.size(1)):
            result.append(self.output(x[:, i, :]))
        out = torch.stack(result, dim=1)
        return out

    def convert(self, x):
        indices = torch.from_numpy(np.arange(0, 28)).type(torch.LongTensor)
        x = torch.cat((torch.zeros(x.size(0), 1, 28), x), dim=1)
        x = torch.index_select(x, 1, indices)
        return x

    def forward(self, x):
        code = self.encoder(x)
        x = self.convert(x)
        out, _ = self.decoder(x, code)
        out = self.output_layer(out)
        return out


autoencoder = AE()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()


def train(epoch):
    for step, (x, _) in enumerate(train_loader):
        b_x = x.view(-1, 28, 28)
        output = autoencoder(b_x)
        loss = loss_func(output, b_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 300 == 0:
            print('EPOCH:', epoch, '|step:', step, '|loss:%.4f' % loss.item())
            # plt.subplot(1, 2, 1)
            # plt.imshow(b_x[0])
            # plt.subplot(1, 2, 2)
            # plt.imshow(output.data[0].numpy())
            # plt.show()


def test(epoch):
    with torch.no_grad():
        h = autoencoder.encoder(test_x[0].view(-1, 28, 28))
        output = []
        out = torch.zeros((1, 1, 28))
        for _ in range(TIME_STEP):
            # print('out:', out)
            # print('h:', h)
            out, h = autoencoder.decoder(out, h)
            out = autoencoder.output_layer(out)
            output.append(out.data.squeeze().numpy())
        output = torch.from_numpy(np.array(output)).view(1, 28, 28)
        loss = loss_func(test_x[0].view(-1, 28, 28), output)

        # output = autoencoder(test_x[0].view(1, 28, 28))
        # loss = loss_func(test_x[0].view(-1, 28, 28), output)

        print('EPOCH:', epoch, '|loss:%.4f' % loss.item())
        plt.subplot(1, 2, 1)
        plt.imshow(test_x[0])
        plt.subplot(1, 2, 2)
        plt.imshow(output.data[0].numpy())
        plt.show()


if __name__ == '__main__':
    for epoch in range(EPOCH):
        train(epoch)
        test(epoch)