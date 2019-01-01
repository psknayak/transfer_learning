#Import required models
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets
import time
import copy
import data_splitting
from data_splitting import habitat_dataset

#Habitat dataset path and destination path of .csv file
dataset_path = './datasets/Habitats/Habitats Dataset'
destination_path = './datasets/Habitats'

#Number of habitat classes
habitat_classes = 24

#Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Create and load train and validation dataframes
dframe, train_df, validation_df = habitat_dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=0.9)

#Load the pre-trained model resnet34 and reset the final fully connected layer
habitat_model = models.resnet34(pretrained=True)
for params in habitat_model.parameters():
    params.requires_grad = True
    num_ftrs = habitat_model.fc.in_features
    habitat_model.fc = nn.Linear(num_ftrs, habitat_classes)
habitat_model = habitat_model.to(device)

#Loss function used -> Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

"""All parameters of the network are being optimized here"""
#Stochastic Gradient Descent is used here with a learning rate set to 0.001 and a momentum of 0.9
optimizer_ft = optim.SGD(habitat_model.parameters(), lr=0.001, momentum=0.9)

"""LR is decayed by a factor of 0.1 every 5 epochs"""
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

def data_loading(train_df, validation_df):
    """ Loads the datasets into the model for training

        Args:
            train_df: training dataframe of habitats dataset
            validation_df: validation dataframe of habitats dataset

        Returns:
            data_loaders: Dict of data loaders of train and validation datasets
            dataset_sizes: Dict of sizes of train and validation datasets
    """
    ###CODE START###

    #Data augmentation and normalization for training
    train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #Only normalization for validation
    validation_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #Apply the respective transforms on training and validation datasets
    train_dataset = habitat_dataset.ImageDataset(train_df, transform=train_transform)
    validation_dataset =  habitat_dataset.ImageDataset(validation_df, transform=validation_transform)

    #Set batch size to 8
    batch_size = 8

    #Create Dataloaders for train and validation datasets
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=4)

    #Create a dict for easy access
    dataloaders = {'train': train_loader,'val':validation_loader}
    dataset_sizes = {'train':len(train_dataset),'val':len(validation_dataset)}

    #stores the names of classes of the habitats as a list
    class_names = train_dataset.classes
    ###CODE END###
    return dataloaders, dataset_sizes

def train_model(model, train_df, validation_df, criterion, optimizer, scheduler, num_epochs=25):
    """Trains the model and saves the best model weights to be loaded the next time the model is to be used. 

        Args:
            model: The habitat model to be trained
            train_df: The training dataframe
            validation_df: The validation dataframe
            criterion: The loss function to be used
            optimizer: To update the model weights in response to the output of the loss function
            scheduler: To Change LR by a certain factor in periodic intervals
            num_epochs: Number of times the dataset goes through the model training stage. Defaults to 25

        Returns:
            model: The trained habitat model
        """
    ##CODE START###
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataloaders, dataset_sizes = data_loading(train_df, validation_df)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation stage
        for stage in ['train', 'val']:
            if stage == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            #Initialize the loss and the correctly predicted labels to zero
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[stage]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # parameter gradients are made zero
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(stage == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training stage
                    if stage == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #calculate loss and accuracy for every epoch and print
            epoch_loss = running_loss / dataset_sizes[stage]
            epoch_acc = running_corrects.double() / dataset_sizes[stage]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                stage, epoch_loss, epoch_acc))

            # deep copy the model
            if stage == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    ###CODE END###
    return model

if __name__ == "__main__":
    habitat_model = train_model(habitat_model, train_df, validation_df, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=20)
    hmod_path = './habitat-model.pth' 
    torch.save(habitat_model, hmod_path)
    pass
