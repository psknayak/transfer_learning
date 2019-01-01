#Import required modules
import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import NeuralNet
from PIL import Image
import argparse
import cv2

def predict_image(image, model):
    """Predicts the output label of a given image through its index

        Args:
            image: The image whose output is to be predicted
            model: The loaded model(animal or habitat) based on which image is inputted by the user

        Returns:
            index: The index of the highest probability in the output layer(softmax layer) of the model
    """
    ###CODE START###

    ###Transformation and Normalization for the test image so that
    ###the dimensions match that of the input layer of the trained model###
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    image_tensor = test_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    input = Variable(image_tensor.cuda())
    output = model(input)
    index = output.data.cuda().argmax()
    ###CODE END###
    return index

"""
    Usage of argparse for command line interfacing
"""
###CODE START FOR ARGPARSE###
parser = argparse.ArgumentParser(description="Using a pre-trained pytorch model to classify animals and another to classify habitats", add_help=False)
parser.add_argument('-a', '--animal', help="Outputs animal name from image")
parser.add_argument('-h', '--habitat', help="Outputs habitat name from image")
parser.add_argument('--amod', default="./animal-model.pth", help="Use a trained animal model")
parser.add_argument('--hmod', default="./habitat-model.pth", help="Use a trained habitat model")
args = parser.parse_args()
###CODE END FOR ARGPARSE###


animal_labels = ['arctic fox', 'bear', 'bee', 'butterfly', 'cat', 'cougar', 'cow', 'coyote', 'crab',
        'crocodile', 'deer', 'dog', 'eagle', 'elephant', 'fish', 'frog', 'giraffe',
        'goat', 'hippo', 'horse', 'kangaroo', 'lion', 'monkey', 'otter', 'panda',
        'parrot', 'penguin', 'raccoon', 'rat', 'seal', 'shark', 'sheep', 'skunk',
        'snake', 'snow leopard', 'tiger', 'yak', 'zebra']

habitat_labels = ['baseball', 'basketball court', 'beach', 'circular farm', 'cloud',
        'commercial area', 'dense residential', 'desert', 'forest', 'golf course', 'harbor', 'island',
        'lake', 'meadow', 'medium residential area', 'mountain', 'rectangular farm', 'river',
        'sea glacier', 'shrubs', 'snowberg', 'sparse residential area', 'thermal power station', 'wetland']

#if user's argument is -a, then load trained animal model
if args.animal:
    model = torch.load(args.amod)
    img = Image.open(args.animal)
    index = predict_image(img, model)
    #Return the value of the animals_labels list as specified by index value
    name = animal_labels[index]
    img.show()
    #print the predicted animal label
    print(name)

#if user's argument is -h, then load trained habitat model
else:
    model = torch.load(args.hmod)
    img = Image.open(args.habitat)
    index = predict_image(img, model)
    #Return the value of the habitats_labels list as specified by index value
    name = habitat_labels[index]
    img.show()
    #print the predicted habitat label
    print(name)

if __name__ == '__main__':
    pass
