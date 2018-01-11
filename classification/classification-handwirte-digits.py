import torch 
from torch.autograd import Variable
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import matplotlib.pyplot as plt


# Hyper Parameters
num_epochs = 5
batch_size = 32
learning_rate = 1e-3


class Net(nn.Module):
	def __init__(self, input_size, num_classes):
		super(Net, self).__init__()
		self.hidden = nn.Sequential(
			nn.Linear(input_size, 512),
			nn.BatchNorm1d(512),
			nn.ReLU()
		)
		self.hidden2 = nn.Sequential(
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU()
		)
		self.out = nn.Sequential(
			nn.Linear(256, num_classes)
		)

	def forward(self, x):
		x = self.hidden(x)
		x = self.hidden2(x)
		x = self.out(x)
		return x


def main():
    # MNIST Dataset
    train_dataset = MNIST(
        root='./data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor()
    )

    # Data Loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    if torch.cuda.is_available():
        net = Net(28*28, 10).cuda()
    else:
        net = Net(28*28, 10)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = Variable(images).cuda(), Variable(labels).cuda()
            else:
                images, labels = Variable(images), Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(images.view(images.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # Test the model
    net.eval()
    correct = total = 0

    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = Variable(images).cuda(), labels.cuda()
        else:
            images = Variable(images)

        outputs = net(images.view(images.size(0), -1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.4f%%' % (100 * correct / total))

    for i_batch, (images, labels) in enumerate(test_loader):
        if i_batch == 5:
            if torch.cuda.is_available():
                images, labels = Variable(images).cuda(), labels.cuda()
            else:
                images = Variable(images)
            outputs = net(images.view(images.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            print(predicted.cpu().numpy())

            plt.figure()
            grid = utils.make_grid(images.cpu().data)
            plt.imshow(grid.numpy().transpose(1, 2, 0))
            plt.axis('off')
            plt.show()


    # Save the Trained Model
    torch.save(net.state_dict(), 'classification.pkl')


if __name__ == '__main__':
    main()