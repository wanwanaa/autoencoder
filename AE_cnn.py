import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt

# hyper Parameters
EPOCH = 1
BATCH_SIZE = 32
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='\mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.MNIST(root='/mnist', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2]/255


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder1 = nn.Sequential( # (1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), # (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2), # (16, 14, 14)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (32, 7, 7)
        )
        self.decoder = nn.Sequential(
            # nn.MaxUnpool2d(2, stride=2), # (32, 14, 14) # indices
            # nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2), # (16, 14, 14)
            # nn.ReLU(),
            # nn.MaxUnpool2d(2, stride=2), # (16, 28, 28)
            # nn.ConvTranspose2d(16, 1, 5, 1, 2), # (1, 28, 28)
            # nn.Sigmoid()
            nn.Linear(32 * 7 * 7, 512),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = x.view(-1, 32*7*7)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)
        return x


autoencoder = AE()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

if __name__ == '__main__':
    fig = plt.figure()
    for epoch in range(EPOCH):
        for step, (b_x, _) in enumerate(train_loader):
            output = autoencoder(b_x)
            loss = loss_func(output, b_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        result = autoencoder(test_x[0].view(1, 1, 28, 28))
        plt.subplot(121)
        plt.imshow(test_x[0].view(28, 28))
        plt.subplot(122)
        plt.imshow(result.data.view(28, 28))
        plt.show()