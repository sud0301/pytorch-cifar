'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import *

import os
import argparse
from config import pr2_config 

#from models import vgg
#from mobilenet import *
from resnet import *
#from vgg import *
#from badgan_net import *
#from googlenet import *
#from utils import progress_bar
from torch.autograd import Variable
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
data_load = 0

def DataLoader(raw_loader, indices, batch_size):
    images, labels = [], []
    for idx in indices:
        image, label = raw_loader[idx]
        images.append(image)
        labels.append(label)

    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels, dtype=np.int64)).squeeze()
    return zip(images, labels)

# Data
print('==> Preparing data..')
'''
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''

if data_load==1:
	trainloader = pickle.load( open( "trainloader.pickle", "rb" ) )
	testloader = pickle.load( open( "testloader.pickle", "rb" ) )
else:

	transform = transforms.Compose([transforms.Resize(size=(32, 32), interpolation=2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	train_labeled_set = ImageFolder('/misc/lmbraid19/mittal/yolo-9000/yolo_dataset/train_labeled/', transform=transform)


	train_labeled_indices = np.arange(len(train_labeled_set))
	np.random.shuffle(train_labeled_indices)
	mask = np.zeros(train_labeled_indices.shape[0], dtype=np.bool)
	labels = np.array([train_labeled_set[i][1] for i in train_labeled_indices], dtype=np.int64)
	for i in range(6):
	    mask[np.where(labels == i)[0][: int(pr2_config.size_labeled_data / 6)]] = True
	    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]

	train_labeled_indices = train_labeled_indices[mask]
	print ('# Labeled indices ', len(train_labeled_indices) )

	test_set = ImageFolder('/misc/lmbraid19/mittal/yolo-9000/yolo_dataset/test_labeled/', transform=transform)
	test_indices = np.arange(len(test_set))
	print ('# Test indices ', len(test_indices))

	tr_images, tr_labels = [], []
	for idx in train_labeled_indices:
	    image, label = train_labeled_set[idx]
	    tr_images.append(image)
	    tr_labels.append(label)
	    images = torch.stack(tr_images, 0)
	    labels = torch.from_numpy(np.array(tr_labels, dtype=np.int64)).squeeze()

	trainset = tuple(zip(images, labels))

	testset = test_set

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=pr2_config.train_batch_size, shuffle=True, num_workers=8)
	testloader = torch.utils.data.DataLoader(testset, batch_size=pr2_config.dev_batch_size, shuffle=False, num_workers=8)

	#trainloader = DataLoader(train_labeled_set, train_labeled_indices, pr2_config.train_batch_size)
	#dev_loader = DataLoader(pr2_config, test_set, test_indices, pr2_config.dev_batch_size)

	#pickle.dump( trainloader, open( "trainloader.pickle", "wb" ) )
	#pickle.dump( testloader, open( "testloader.pickle", "wb" ) )



classes = ('bottle', 'chair', 'cup', 'display', 'keyboard', 'table')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    #net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    #net = GoogLeNet()
    #net = DenseNet121()
    # net = ResNeXt29_2x64d()
    #net = MobileNet()
    #net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    #net = BadGAN(pr2_config)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()        
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
           # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        #print ('Loss: %.3f | Acc: %.3f%% (%d/%d)' (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print ('Loss: ' + str(train_loss/(batch_idx+1)) + ' | ' + "Acc: " + str(100.*correct/total) + ' ' +  str(correct)+ '/' + str(total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
           # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        #print ('Loss: %.3f | Acc: %.3f%% (%d/%d)' (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print ('Loss: ' + str(test_loss/(batch_idx+1)) + ' | ' + "Acc: " + str(100.*correct/total) + ' ' +  str(correct)+ '/' + str(total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
        #    os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+300):
    train(epoch)
    test(epoch)
