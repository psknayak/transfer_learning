from __future__ import print_function,division
import utils
from utils import animal_dataset,habitat_dataset
import model_init, model
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import time
import copy
import argparse
import cv2

def predict_image(image, model):
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_tensor = test_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    input = Variable(image_tensor.cuda())
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

parser = argparse.ArgumentParser(description="Training a pytorch model to classify animals and another to classify habitats",add_help=False)
parser.add_argument('-a','--animal',help="Outputs animal name from image")
parser.add_argument('-h','--habitat',help="Outputs habitat name from image")
parser.add_argument('--amod',default= "./animal-model.pth",help="Use a trained animal model")
parser.add_argument('--hmod',default="./habitat-model.pth",help="Use a trained habitat model")
args = parser.parse_args()

animal_labels=['arctic fox', 'bear', 'bee', 'butterfly', 'cat', 'cougar', 'cow', 'coyote', 'crab',
        'crocodile', 'deer', 'dog', 'eagle', 'elephant', 'fish', 'frog', 'giraffe',
        'goat', 'hippo', 'horse', 'kangaroo', 'lion', 'monkey', 'otter', 'panda',
        'parrot', 'penguin', 'raccoon', 'rat', 'seal', 'shark', 'sheep', 'skunk',
        'snake', 'snow leopard', 'tiger', 'yak', 'zebra']
habitat_labels=['baseball', 'basketball court', 'beach', 'circular farm', 'cloud', 'commercial area',
        'dense residential','desert','forest','golf course','harbor','island',
        'lake','meadow','medium residential area','mountain','rectangular farm','river',
        'sea glacier','shrubs','snowberg','sparse residential area','thermal power station','wetland']

if args.animal:
    model=torch.load(args.amod)
    img = Image.open(args.animal)
    index = predict_image(img, model)
    name = animal_labels[index]
    img.show()
    print("Image is a {}".format(name))
else:
    model=torch.load(args.hmod)
    img = Image.open(args.habitat)
    index = predict_image(img, model)
    name = habitat_labels[index]
    img.show()
    print("Image is a {}".format(name))

if __name__ == '__main__':
    pass

