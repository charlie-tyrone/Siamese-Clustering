
import os
import torch                                                                                
import numpy as np                                                                          
import matplotlib.pyplot as plt
import torch.utils.data as tech 

from utils import *
from PIL import Image                                                                       
from tqdm import tqdm

#-------------------------------------
# Initialize: All Siamese Datasets 
#-------------------------------------

class Siamese_Dataset(tech.Dataset):
    
    def __init__(self, params, choice):
      
        #-----------------------------
        # Load: All Dataset Paramters 
        #-----------------------------

        self.choice = choice 
        self.data = params['data']                                                          
        self.dataset = params['dataset']                                                    
        self.transforms = params['transforms']
        self.in_channels = params['input_size']

        if(choice.lower() == 'train'):
            self.num_samples = params['num_train_samples']
        else:
            self.num_samples = params['num_valid_samples']

        #-----------------------------
        # Create: Non-Custom Dataset
        # TODO: Support Custom Dataset 
        #-----------------------------

        if(self.dataset != 1): 
            self.classes = params['classes']                                                
            self.get_two_classes()                                                          
            
            if(self.choice.lower() == 'train' or self.choice.lower() == 'valid'):
                self.make_siamese_dataset()
            else:
                self.load_anchors(params['path_anchors'])
                self.make_test_dataset()
 
    #---------------------------------
    # Load: KNN Prototype(s), Folder 
    #---------------------------------
             
    def load_anchors(self, path_anchors):

        self.anchors = []
        self.anchor_labels = []
        
        all_classes = os.listdir(path_anchors)
        
        for count, current_class in enumerate(all_classes):
        
            current_class = os.path.join(path_anchors, current_class)
            all_files = os.listdir(current_class) 
            all_files.sort()
        
            for anchor_file in all_files:
                anchor_file = os.path.join(current_class, anchor_file) 
           
                if(self.in_channels == 1):
                    image = Image.open(anchor_file).convert('L')
                    image = torch.tensor(np.asarray(image))
                    image = torch.unsqueeze(image, dim=0)

                elif(self.in_channels == 3): 
                    image = Image.open(anchor_file).convert('RGB')
                    image = torch.tensor(np.swapaxes(np.asarray(image), 0, 2))
                    image = image.type('torch.FloatTensor')
                else:
                    print('\nError: Only Can Transform Greyscale or RGB Images\n')
                    exit()                

                image = torch.unsqueeze(image, dim=0).float()
                image = data_transforms(image, self.transforms) 
                image = torch.squeeze(image, dim=0)
      
                self.anchors.append(image)
                self.anchor_labels.append(count)

    #---------------------------------
    # Load: 2 Class From N Class Data 
    #---------------------------------

    def get_two_classes(self):
        
        dataset_all = {}
        dataset = self.data
        class_a, class_b = self.classes

        dataset.data = dataset.data.numpy()
        dataset.targets = dataset.targets.numpy()

        dataset_idxs_a = np.where(dataset.targets == class_a)
        dataset_idxs_b = np.where(dataset.targets == class_b)

        dataset_all['c0'] = torch.tensor(dataset.data[dataset_idxs_a]) 
        dataset_all['c1'] = torch.tensor(dataset.data[dataset_idxs_b])

        self.siamese_data = dataset_all

    #---------------------------------
    # Initialize: Siamese Test Dataset
    #---------------------------------

    def make_test_dataset(self):

        choice = self.choice
        dataset = self.siamese_data
        num_samples = self.num_samples
        
        siamese_dataset = []
        data_c0, data_c1= dataset['c0'], dataset['c1']
        
        count = 0
        name = choice + ' similar'
        pbar = tqdm(total=num_samples, desc=name) 
        while(count < num_samples): 
            for data_sample in data_c0:
                if(count == num_samples):
                    break
                
                siamese_dataset.append([data_sample, 0])
                count = count + 1
                pbar.update(1)

        pbar.close()

        count = 0
        name = choice + ' different'
        pbar = tqdm(total=num_samples, desc=name) 
        while(count < num_samples): 
            for data_sample in data_c1:
                if(count == num_samples):
                    break
                
                siamese_dataset.append([data_sample, 1])
                count = count + 1
                pbar.update(1)

        pbar.close()

        self.siamese_data = siamese_dataset

    #---------------------------------
    # Initialize: Siamese Pair Dataset
    # Supports: Only 2 Classes, TODO
    #---------------------------------

    def make_siamese_dataset(self):
    
        choice = self.choice
        dataset = self.siamese_data
        num_samples = self.num_samples
        
        siamese_dataset = []
        data_c0, data_c1= dataset['c0'], dataset['c1']
 
        #-----------------------------
        # Initialize: Similar Class
        #-----------------------------
       
        count = 0
        name = choice + ' similar'
        pbar = tqdm(total=num_samples, desc=name) 
        while(count < num_samples): 
            for data_sample in data_c0:
                if(count == num_samples):
                    break
                
                index = np.random.randint(num_samples)
                similar_sample = data_c0[index]
                
                siamese_dataset.append([data_sample, similar_sample, 0])
                count = count + 1
                pbar.update(1)

        pbar.close()

        #-----------------------------
        # Initialize: Dissimilar Class
        #-----------------------------

        count = 0
        name = choice + ' different'
        pbar = tqdm(total=num_samples, desc=name) 
        while(count < num_samples): 
            for data_sample in data_c1:
                if(count == num_samples):
                    break
                
                index = np.random.randint(num_samples)
                different_sample = data_c0[index]
                
                siamese_dataset.append([data_sample, different_sample, 1])
                count = count + 1
                pbar.update(1)

        pbar.close()         
        
        self.siamese_data = siamese_dataset  

    #---------------------------------
    # Siamese: Input(s) =  Pairs
    # Examples: [TT, TT] OR [TT, FA]
    #---------------------------------

    def __getitem__(self,index):
        
        #-----------------------------
        # Inputs: Model, Train / Valid 
        #-----------------------------

        if(self.choice.lower() == 'train' or self.choice.lower() == 'valid'):        
            sample_1, sample_2, label = self.siamese_data[index]
            if( len(sample_1.size()) == 2 ):
                sample_1 = torch.unsqueeze(sample_1, 0)
                sample_2 = torch.unsqueeze(sample_2, 0)
            return sample_1, sample_2, label

        #-----------------------------
        # Inputs: Model, All Testing 
        #-----------------------------

        else:
            sample, label = self.siamese_data[index]
            if( len(sample.size()) == 2 ):
                sample = torch.unsqueeze(sample, 0)
            self.anchor_labels = np.asarray(self.anchor_labels)
            return self.anchors, self.anchor_labels, sample, label
 
    #---------------------------------
    # Siamese: Number, Dataset Samples
    #---------------------------------

    def __len__(self):
        
        return len(self.siamese_data)        
