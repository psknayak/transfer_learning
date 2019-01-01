# import required libraries
from PIL import Image
import numpy as np
import sys
import os
import pandas as pd
import csv
import torch, torchvision
from torch.utils.data.dataset import Dataset
import cv2


def create_meta_csv(dataset_path, destination_path):
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'animal_dataset.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images required are loaded randomly but on demand using their paths.

    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'animal_dataset.csv' was created successfully else returns an exception
    """

    # Change dataset path accordingly
    DATASET_PATH = os.path.abspath(dataset_path)
    print(DATASET_PATH)
    if not os.path.exists(os.path.join(DATASET_PATH, "/animal_dataset.csv")):
        filelist = os.listdir(DATASET_PATH)
        dataSet = list()
        for x in filelist:
            myDir = DATASET_PATH + '/{}'.format(x)

            # Useful function
            def createFileList(myDir, format='.jpg'):
                fileList = []
                for root, dirs, files in os.walk(myDir, topdown=False):
                    for name in files:
                        if name.endswith(format):
                            fullName = os.path.join(root, name)
                            fileList.append(fullName)
                return fileList

            myFileList = createFileList(myDir)
            for file in myFileList:
                dataSet.append([file, x])

        # change destination_path to DATASET_PATH if destination_path is None

        if destination_path == None:
            destination_path = dataset_path

        # write out as animal_dataset.csv in destination_path directory
        dp = destination_path + '/animal_dataset.csv'
        with open(dp, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'label'])
            for i in dataSet:
                writer.writerow(i)
        print('done')
        # if no error
        return True


def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'animal_dataset.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a
    fraction in split parameter.

    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """
    if create_meta_csv(dataset_path, destination_path=destination_path):
        dframe = pd.read_csv(os.path.join(destination_path, 'animal_dataset.csv'))

    # shuffle if randomize is True or if split specified and randomize is not specified
    # so default behavior is split
    if randomize == True or (split != None and randomize == None):
        # shuffle the dataframe here
        dframe = dframe.sample(frac=1).reset_index(drop=True)
        pass

    if split != None:
        train_set, test_set = train_test_split(dframe, split)
        return dframe, train_set, test_set

    return dframe


def train_test_split(dframe, split_ratio):
    """Splits the dataframe into train and test subset dataframes.

    Args:
        split_ration (float): Divides dframe into two splits.

    Returns:
        train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
        test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
    """
    # divide into train and test dataframes
    train_data = dframe[:int(len(dframe) * split_ratio)]
    test_data = dframe[int(len(dframe) * split_ratio):]
    return train_data, test_data


class ImageDataset(Dataset):
    """Image Dataset that works with images

    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.

    Examples
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.classes = data['label'].unique()
        self.label_encode = {'arctic fox': 0, 'bear': 1, 'bee': 2, 'butterfly': 3, 'cat': 4, 'cougar': 5,'cow': 6,'coyote': 7,'crab': 8,'crocodile': 9,'deer': 10,'dog': 11,'eagle': 12,'elephant': 13,'fish': 14,'frog': 15,'giraffe': 16,'goat': 17,'hippo': 18,'horse': 19,'kangaroo': 20,'lion': 21,'monkey': 22,'otter': 23,'panda': 24,'parrot': 25,'penguin': 26,'raccoon': 27,'rat': 28,'seal': 29,'shark': 30,'sheep': 31,'skunk': 32,'snake': 33,'snow leopard': 34,'tiger': 35,'yak': 36,'zebra': 37}
        # self.label_encode = {'baseball': 0, 'basketball court': 1, 'beach': 2, 'circular farm': 3, 'cloud': 4, 'commercial area': 5,'dense residential': 6,'desert': 7,'forest': 8,'golf course': 9,'harbor': 10,'island': 11,'lake': 12,'meadow': 13,'medium residential area': 14,'mountain': 15,'rectangular farm': 16,'river': 17,'sea glacier': 18,'shrubs': 19,'snowberg': 20,'sparse residential area': 21,'thermal power station': 22,'wetland': 23}
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        img = Image.open(img_path).convert('RGBA')
        bkgnd = Image.new('RGBA', img.size, (255,255,255))
        alpha_compose = Image.alpha_composite(bkgnd,img)
        image = alpha_compose.convert('RGB')
        label = self.label_encode[self.data.iloc[idx]['label']]  # get label (derived from self.classimport torch.utils.data.Datasetes; type: int/long) of image
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    # test config
    dataset_path = './datasets/Animals/Animals Dataset'
    dest = './datasets/Animals/'
    classes = 38
    total_rows = 4131
    randomize = True
    clear = True

    # test_create_meta_csv()
    # df, trn_df, val_df, tst_df = create_and_load_meta_csv_df(dataset_path, destination_path=dest, randomize=True,
    #                                                 split=0.99)
                                                 