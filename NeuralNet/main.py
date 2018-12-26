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

# dataset_path = './datasets/Animals/AnimalsDataset'
# destination_path = './datasets/Animals'
# # dataset_path = './datasets/Animals/AnimalsDataset'
# # destination_path = './datasets/Animals'
# # # animals_path = './datasets/Animals/AnimalsDataset'
# # # animals_dest_path = './datasets/Animals'
# # # habitats_path = './datasets/Habitats/HabitatsDataset'
# # # habitats_dest_path = './datasets/Habitats'
# amodel_name = "resnet101"
# hmodel_name = "resnet152"
# animal_classes = 38
# habitat_classes = 24

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dframe, train_df, validation_df = animal_dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=0.9)
# # dframe, train_df, validation_df = habitat_dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=0.9)


# animal_model, input_size = model_init.initialize_model(amodel_name, animal_classes, use_pretrained=True)
# animal_model = animal_model.to(device)

# # habitat_model, input_size = model_init.initialize_model(hmodel_name, habitat_classes, use_pretrained=True)
# # habitat_model = habitat_model.to(device)

# criterion = nn.CrossEntropyLoss()

# """Observe that all parameters are being optimized"""
# optimizer_ft = optim.SGD(animal_model.parameters(), lr=0.001, momentum=0.9)
# # optimizer_ft = optim.SGD(habitat_model.parameters(), lr=0.001, momentum=0.9)

# """Decay LR by a factor of 0.1 every 7 epochs"""
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# animal_model = animal_model.train_model(animal_model, train_df, validation_df, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
# habitat_model = habitat_model.train_model(habitat_model, train_df, validation_df, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=20)


# amod_path = './animal-model.pth' 
# torch.save(animal_model, amod_path)
# print("[Loading the trained model on the datasets]")

# hmod_path = './habitat-model.pth'
# torch.save(habitat_model, hmod_path)
# print("[Loading the trained model on the datasets]")

# img2 = args.habitat
# test_transform = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# # habitat_model=torch.load(args.hmod)
# # habitat_model.eval()
def predict_image(image):
    model=torch.load('./animal-model.pth')
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

def main():
    parser = argparse.ArgumentParser(description="Training a pytorch model to classify animals and another to classify habitats")
    parser.add_argument('-a','--animal', default='./6620.jpg',help="Outputs animal name from image")
    parser.add_argument('-H','--habitat',help="Outputs animal name from image")
    parser.add_argument('-amodel','--amod',default="./animal-model.pth",help="Use a trained animal model")
    parser.add_argument('-hmodel','--hmod',default="./habitat-model.pth",help="Use a trained habitat model")
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
    
    img = Image.open(args.animal)
    index = predict_image(img)
    name = animal_labels[index]
    print(name)

if __name__ == '__main__':
    main()
