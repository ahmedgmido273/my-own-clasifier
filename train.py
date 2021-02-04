import time
import torch
import json
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms
from torchvision import models as models

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help="input the data dir to train the model")
parser.add_argument('--save_dir', help="input the format to save the model", default="checkpoint.pth")
parser.add_argument('--arch', default="vgg11", help="enter the pretrained model u want to work with")
parser.add_argument('--learning_rate', default=0.001, help="enter the learning rate for the optmiers")
parser.add_argument('--hidden_units', default=4096, help="number of units in the hidden layer")
parser.add_argument('--epochs', default=3 ,type=int , help="number of epochs you wan to train with")
parser.add_argument('--gpu', default="cuda", help="enter cuda if u want to train on a gpu or cpu if u don't")
args = parser.parse_args()



data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.gpu
input_size = 25088
print_every = 50
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
#transforms to the data
transforms_list = {'train': [transforms.RandomRotation(30),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                  'test_val': [transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]}
# the data 
train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose(transforms_list['train']))
test_data = datasets.ImageFolder(valid_dir, transform=transforms.Compose(transforms_list['test_val']))
val_data = datasets.ImageFolder(test_dir, transform=transforms.Compose(transforms_list['test_val']))
# the loaders for the data
train_load = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_load  = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
valid_load = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def getTheModel(arch, hidden_units, input_size=25088, output_size=102):
    #models = {'vgg11': models.vgg11(pretrained=True),
     #         'vgg13': models.vgg13(pretrained=True),
      #        'vgg16': models.vgg16(pretrained=True),
       #       'vgg19': models.vgg19(pretrained=True)}
    #model = models[arch]
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        
        
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                       ('relu',nn.ReLU()),
                                       ('dropout', nn.Dropout(0.2)),
                                       ('fc2', nn.Linear(hidden_units, output_size)),
                                       ('output', nn.LogSoftmax(dim=1))]))
                                    
    
    model.classifier = classifier
    return model

def train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_accuracy = check_validation_set(validloader,device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy,4)))

                running_loss = 0
    print("DONE TRAINING!")
    
def check_validation_set(valid_loader,device='cpu'):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total



model = getTheModel(arch, hidden_units)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
train_model(model, train_load, valid_load, epochs, print_every, criterion, optimizer, device)

def check_accuracy_on_test(testloader,device='cpu'):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return correct / total


check_accuracy_on_test(test_load,device)


model.class_to_idx = train_data.class_to_idx
model.cpu()
checkpoint ={'arch': arch,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir)












