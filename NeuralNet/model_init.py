import torch
import torch.nn as nn
import torchvision
from torchvision import models


def initialize_model(model_name,num_classes,use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        for params in model_ft.parameters():
            params.requires_grad = True
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        for params in model_ft.parameters():
            params.requires_grad = True
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    else:
        print("Invalid Model Name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
# model_name = "resnet18"
# num_classes = 38
# model_ft, input_size = initialize_model(model_name, num_classes, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)

if __name__ == "__main":
    pass
