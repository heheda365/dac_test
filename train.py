import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar
import skynet

import dataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((320, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_data_path = 'data_training'
trainset = dataset.CustomDataset(train_data_path, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
# net = VGG('VGG16')
net = skynet.SkyNet()

net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters())

def iou(bbox0, bbox1):
    minx0 = bbox0[0] - bbox0[2] / 2
    maxx0 = bbox0[0] + bbox0[2] / 2
    miny0 = bbox0[1] - bbox0[3] / 2
    maxy0 = bbox0[1] + bbox0[3] / 2

    minx1 = bbox1[0] - bbox1[2] / 2
    maxx1 = bbox1[0] + bbox1[2] / 2
    miny1 = bbox1[1] - bbox1[3] / 2
    maxy1 = bbox1[1] + bbox1[3] / 2

    left_column_max  = max(minx0, minx1)
    right_column_min = min(maxx0, maxx1)
    up_row_max       = max(miny0, miny1)
    down_row_min     = min(maxy0, maxy1)
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = bbox0[2] * bbox0[3]
        S2 = bbox1[2] * bbox1[3]
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(outputs)
        # print('===============')
        # print(targets)
        # out_x = outputs[:, 0]
        # out_y = outputs[:, 1]
        # out_w = outputs[:, 2]
        # out_h = outputs[:, 3]

        # out_x = F.sigmoid(out_x)
        # out_y = F.sigmoid(out_y)
        # out_w = 
        outputs = torch.sigmoid(outputs)


        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        miou = 0
        for i in range(len(targets)):
            miou += iou(outputs[i], targets[i])
        miou /= 64


        progress_bar(batch_idx, len(trainloader), 'Loss: %.8f | iou： %.4f' 
            % (train_loss/(batch_idx+1), miou))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
