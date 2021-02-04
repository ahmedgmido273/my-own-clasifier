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
parser.add_argument('checkpoint')
parser.add_argument('--top_k', type=int,default=5)
parser.add_argument('--category_names',default="cat_to_name.json" )
parser.add_argument('--gpu', default="cuda")
args = parser.parse_args()

checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
device = args.gpu
with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'        
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

        
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model = models.vgg11(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                       ('relu',nn.ReLU()),
                                       ('dropout', nn.Dropout(0.3)),
                                       ('fc2', nn.Linear(4096, 102)),
                                       ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    
    # Edit
    trans_image = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    # Dimension
    image_proced = trans_image(pil_image).numpy().transpose((0, 2, 1))
    return image_proced
  

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model = load_model(model)
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        return probs[0].tolist(), classes[0].add(1).tolist()
    flowers_list = [cat_to_name[str(index + 1)] for index in np.array(p[1][0])]
   
    return score, flowers_list



def display_prediction(image_path,model):
    probs, classes = predict(image_path,model)
    plant_classes = [cat_to_name[str(cls)] + "({})".format(str(cls)) for cls in classes]
    im = Image.open(image_path)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im);
    y_positions = np.arange(len(plant_classes))
    ax[1].barh(y_positions,probs,color='blue')
    ax[1].set_yticks(y_positions)
    ax[1].set_yticklabels(plant_classes)
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_xlabel('Accuracy (%)')
    ax[0].set_title('Top 5 Flower Predictions')
    return None



display_prediction('flowers/train/1/image_06734.jpg','checkpoint.pth')
