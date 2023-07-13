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


from image_handling import load_data
from get_arguments import get_arguments_train

#getting the labels
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

in_args = get_arguments_train()

td_class_to_id, trainloader, testloader, validloader = load_data(in_args.data_directory)
    
model = getattr(models, in_args.arch)(pretrained=True)
    
for p in model.parameters():
    p.requires_grad = False
        
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, in_args.hidden_units)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(in_args.hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))
    
model.classifier = classifier

device = torch.device("cuda" if in_args.gpu else "cpu")
model.to(device);

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    
running_loss = 0
epochs = in_args.epochs
curr_step = 0
printing_step = 20

for epoch in range(epochs):
    
    for inputs, labels in trainloader:
        
        curr_step += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if curr_step%printing_step == 0:

            val_loss = 0
            val_accuracy = 0
            model.eval()

            with torch.no_grad():

                for inputs_1, labels_1 in validloader:

                    inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)

                    logps = model.forward(inputs_1)
                    batch_loss = criterion(logps, labels_1)

                    val_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels_1.view(*top_class.shape)
                    val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/printing_step:.3f}.. "
                          f"Validation loss: {val_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {val_accuracy/len(validloader):.3f}")

                    running_loss = 0
                    model.train()
 
print("-----------------------------------------------")

test_loss = 0
accuracy = 0
model.eval()

for inputs, labels in testloader:
    inputs, labels = inputs.to(device), labels.to(device)
    logps = model.forward(inputs)
    batch_loss = criterion(logps, labels)

    test_loss += batch_loss.item()

    
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader):.3f}")

print("-----------------------------------------------")

if in_args.save_dir == "save_directory":
    model.class_to_idx = td_class_to_id

    torch.save({
                'epochs': epochs,
                'model_name':in_args.arch,
                'hidden_units': in_args.hidden_units,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 'checkpoint.pth')
    
    print("Successfully saved checkpoint")



