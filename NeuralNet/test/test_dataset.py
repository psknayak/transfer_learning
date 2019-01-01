from context import utils
from utils import animal_dataset
import os
import unittest

class TestDatasetModule(unittest.TestCase):
    def setUp(self):
        # simple test
        self.dataset_path = 'datasets/Animals/AnimalsDataset'
        self.dest = 'datasets/Animals/'
        self.clear = False
        self.randomize = True
        self.split = 0.8

    def test_create_meta_csv(self):

        # remove if exists
        if os.path.exists(os.path.join(self.dest, 'animal_dataset.csv')):
            os.remove(os.path.join(self.dest, 'animal_dataset.csv'))
        
        # run function
        created = animal_dataset.create_meta_csv(self.dataset_path, destination_path=self.dest)

        # check function output (should be true on success)
        self.assertTrue(created)

        # check if file generated and delete it if clear is true
        self.assertTrue(os.path.exists(os.path.join(self.dest, 'animal_dataset.csv')))

        if os.path.exists(os.path.join(self.dest, 'animal_dataset.csv')):
            print("Successfully created 'animal_dataset.csv file'")

            if self.clear:
                print('Cleaning now...')
                os.remove(os.path.join(self.dest, 'animal_dataset.csv'))


    def test_create_and_load_meta_csv_df(self):
        # remove if exists
        if os.path.exists(os.path.join(self.dest, 'animal_dataset.csv')):
            os.remove(os.path.join(self.dest, 'animal_dataset.csv'))
        
        # run function
        df, train_df, test_df = animal_dataset.create_and_load_meta_csv_df(self.dataset_path, destination_path=self.dest, randomize=self.randomize, split=self.split)

        self.assertEqual(len(train_df['label'].unique()), len(test_df['label'].unique()))
        
        #  check df size 
        self.assertEqual(len(df), len(train_df) + len(test_df))


        # check if file generated and delete it if clear is true
        self.assertTrue(os.path.exists(os.path.join(self.dest, 'animal_dataset.csv')))

        if os.path.exists(os.path.join(self.dest, 'animal_dataset.csv')):
            print("Successfully created 'animal_dataset.csv file'")

            if self.clear:
                print('Cleaning now...')
                os.remove(os.path.join(self.dest, 'animal_dataset.csv'))

if __name__ == '__main__':
    unittest.main()