import utils
from utils import animal_dataset
import model_init
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

dataset_path = './datasets/Animals/AnimalsDataset'
destination_path = './datasets/Animals'

amodel_name = "resnet101"
animal_classes = 38

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dframe, train_df, validation_df = animal_dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=0.9)

animal_model, input_size = model_init.initialize_model(amodel_name, animal_classes, use_pretrained=True)
animal_model = animal_model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(animal_model.parameters(), lr=0.001, momentum=0.9)

"""Decay LR by a factor of 0.1 every 5 epochs"""
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

def data_loading(train_df, validation_df):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = animal_dataset.ImageDataset(train_df, transform=train_transform)
    validation_dataset =  animal_dataset.ImageDataset(validation_df, transform=validation_transform)

    batch_size = 8

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader,'val':validation_loader}
    dataset_sizes = {'train':len(train_dataset),'val':len(validation_dataset)}
    class_names = train_dataset.classes
    return dataloaders, dataset_sizes

def train_model(model, train_df, validation_df, criterion, optimizer, scheduler, num_epochs=25):
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

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[stage]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
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

            epoch_loss = running_loss / dataset_sizes[stage]
            epoch_acc = running_corrects.double() / dataset_sizes[stage]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                stage, epoch_loss, epoch_acc))

            # deep copy the model
            if stage == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    animal_model = train_model(animal_model, train_df, validation_df, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
    amod_path = './animal-model.pth' 
    torch.save(animal_model, amod_path)
