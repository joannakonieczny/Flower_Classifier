import json
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
import torch
from torch import nn
from collections import OrderedDict
from torch import optim

def load_model(filepath):
    
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['model_name'])(pretrained=True)
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer