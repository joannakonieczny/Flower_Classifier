import PIL
from PIL import Image
from torchvision import datasets, transforms, models
import torch
import os
import numpy as np

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_and_valid_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_and_valid_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_and_valid_transforms)

    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return train_data.class_to_idx, trainloader, testloader, validloader


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #creating the thumbnail
    path = os.path.abspath(os.getcwd())
    with Image.open(path+image) as im:
        im.thumbnail((256, 256))

        #cropping the center
        left = 16
        up = 16
        right = left + 224
        down = up + 224
        pil_image = im.crop((left, up, right, down))

        #color channels + normalizing the image
        np_image = np.array(pil_image)/255
        mean = np.array([0.485, 0.456, 0.406])
        stdev = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean)/stdev

        #transposition
        np_image = np_image.transpose((2,0,1))
        
        np_image = torch.tensor(np_image)
        
    return np_image


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