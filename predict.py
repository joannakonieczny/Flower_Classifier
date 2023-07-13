import json
import ast
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np


from image_handling import *
from get_arguments import get_arguments_predict
from model_handling import load_model

in_args = get_arguments_predict()

#getting the labels
with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)

model, optimizer = load_model(in_args.checkpoint)

device = torch.device("cuda" if in_args.gpu else "cpu")
model.to(device);

model.eval()

image = process_image(in_args.path)
with torch.no_grad():
    output = model.forward(image)
    
ps = torch.exp(output)
    
probs, classes = torch.topk(ps, in_args.topk)


imshow(image)
print("Probability")
for prob, c in probs, classes:
    print(cat_to_name[c], ":        ", prob)