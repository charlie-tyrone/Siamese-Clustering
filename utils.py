import os
import torch
import subprocess
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets

#-------------------------------------
# Initialize: Folder, Model Results 
# Note: Removes Previous Trial Saves
#-------------------------------------

def initialize_folder(path_save):

    if(os.path.exists(path_save) == False):
        subprocess.call(['mkdir', '-p', path_save])
    else:
        subprocess.call(['rm', '-r', path_save]) 
        subprocess.call(['mkdir', '-p', path_save])

#-------------------------------------
# Visualization: Hand-Pick Prototypes 
#-------------------------------------

def plot_all_train(data, label, path_save='../Temp_Anchors'):

    path_save = os.path.join(path_save, str(label)) 
 
    if(os.path.exists(path_save) == False):
        subprocess.call(['mkdir', '-p', path_save])
    else:
        subprocess.call(['rm', '-r', path_save]) 
        subprocess.call(['mkdir', '-p', path_save])

    for count, sample in enumerate(tqdm(data, desc='Saving '+str(label))):

        sample = torch.squeeze(sample)
        sample = np.swapaxes(sample.numpy(), 0, 2)
        
        new_path = os.path.join(path_save, str(count)+'.png') 
        plt.imsave(new_path, sample)

#-------------------------------------
# Visualization: Train Samples Pairs 
#-------------------------------------

def plot_train(train):

    fig, ax = plt.subplots(1, 2)
    for count, (imgs_1, imgs_2, labels) in enumerate(train):
       
        imgs_1 = imgs_1.numpy()
        imgs_2 = imgs_2.numpy()
        labels = labels.numpy()

        for i, (img1, img2, truth) in enumerate(zip(imgs_1, imgs_2, labels)):

            name = 'Similar' if(truth == 0) else 'Different'
            
            fig.suptitle(name)

            img1 = np.squeeze(img1)
            img2 = np.squeeze(img2)
            
            if(len(img1.shape) != 2):
                img1 = np.swapaxes(img1, 0, 2)
                img2 = np.swapaxes(img2, 0, 2)

            '''
            if(truth == 1):
                plt.imsave(str(i+1)+'.png', img1)
                plt.imsave(str(i+2)+'.png', img2)
                input('Images Saved...!')
            '''
 
            ax[0].imshow(img1)
            ax[1].imshow(img2)


            ax[0].set_title('Input 1')
            ax[1].set_title('Input 2')

            plt.show(block=False)
            plt.pause(1)



#-------------------------------------
# Initialize: Basic Data Augmentations 
#-------------------------------------

def data_transforms(data, transforms, data_bit=2**8):
 
    resize = transforms['resize']
    tensor = transforms['tensor']
 
    if(resize[0]): 

        if(torch.is_tensor(data) ==  False):
            data = torch.tensor(data).type('torch.FloatTensor')

        if(len(data.shape) != 4):
            print('\nError: Transform Resize, Requires NCHW Format\n')
            exit()
       
        data = F.interpolate(data, size=resize[1], mode='bilinear', align_corners=False) 
        
    if(tensor):
   
        if(torch.is_tensor(data)):
            data = data / (data_bit - 1)
        else:
            data = torch.tensor(data / (data_bit - 1))

    return data

#-------------------------------------
# Load: Datasets (Train, Valid, Test)
#-------------------------------------

def load_dataset(params, channel_size=1, name='test'):

    dataset = params['dataset']

    if(dataset == 1):

        print('\n#--------------------------')
        print('# Loading: Custom Dataset ')
        print('#--------------------------\n')
    
        path_train = params['path_custom_train'] 
        path_valid = params['path_custom_valid']
        train = datasets.ImageFolder(root = path_train)
        valid = datasets.ImageFolder(root = path_valid)
        test = datasets.ImageFolder(root = path_valid)

        print('Loading -- Complete ')

    elif(dataset == 2):

        print('\n#--------------------------')
        print('# Loading: MNIST Dataset ')
        print('#--------------------------\n')

        name = 'mnist'
        path_train = params['path_mnist_train'] 
        path_valid = params['path_mnist_valid']
        train = datasets.MNIST( root = path_train, train = True, download = True)
        valid = datasets.MNIST( root = path_valid, train = False, download = True)
        test = datasets.MNIST( root = path_valid, train = False, download = True)

        train.data = torch.unsqueeze(train.data, axis=1).float()
        valid.data = torch.unsqueeze(valid.data, axis=1).float()
        test.data = torch.unsqueeze(test.data, axis=1).float()

    elif(dataset == 3):

        print('\n#--------------------------')
        print('# Loading: F-MNIST Dataset ')
        print('#--------------------------\n')

        name = 'fmnist'
        path_train = params['path_fmnist_train'] 
        path_valid = params['path_fmnist_valid']
        train = datasets.FashionMNIST( root = path_train, train = True, download = True)
        valid = datasets.FashionMNIST( root = path_valid, train = False, download = True)
        test = datasets.FashionMNIST( root = path_valid, train = False, download = True)

    elif(dataset == 4):

        print('\n#--------------------------')
        print('# Loading: Cifar10 Dataset ')
        print('#--------------------------\n')

        name = 'cifar'
        path_train = params['path_cifar_train'] 
        path_valid = params['path_cifar_valid']

        train = datasets.CIFAR10( root = path_train, train = True, download = True)
        valid = datasets.CIFAR10( root = path_valid, train = False, download=True)
        test = datasets.CIFAR10( root = path_valid, train = False, download=True)

        train.data = torch.tensor(np.swapaxes(train.data, 1, 3)).type('torch.FloatTensor')
        train.targets = torch.tensor(train.targets)
 
        valid.data = torch.tensor(np.swapaxes(valid.data, 1, 3)).type('torch.FloatTensor')
        valid.targets = torch.tensor(valid.targets)

        test.data = torch.tensor(np.swapaxes(test.data, 1, 3)).type('torch.FloatTensor')
        test.targets = torch.tensor(test.targets)
   
        channel_size = 3

    transforms = params['transforms']
    train.data = data_transforms(train.data, transforms)
    valid.data = data_transforms(valid.data, transforms)
    test.data = data_transforms(test.data, transforms)

    print('Loading -- Complete\n')
    
    return train, valid, test, channel_size, name

#-------------------------------------
# Load: Dataset, Only Train Data
#-------------------------------------

def load_train_dataset(params, channel_size=1, name='test'):

    dataset = params['dataset']

    if(dataset == 1):

        print('\n#--------------------------')
        print('# Loading: Custom Dataset ')
        print('#--------------------------\n')
    
        path_train = params['path_custom_train'] 
        train = datasets.ImageFolder(root = path_train)

    elif(dataset == 2):

        print('\n#--------------------------')
        print('# Loading: MNIST Dataset ')
        print('#--------------------------\n')

        name = 'mnist'
        path_train = params['path_mnist_train'] 
        train = datasets.MNIST( root = path_train, train = True, download = True)

        train.data = torch.unsqueeze(train.data, axis=1).float()

    elif(dataset == 3):

        print('\n#--------------------------')
        print('# Loading: F-MNIST Dataset ')
        print('#--------------------------\n')

        name = 'fmnist'
        path_train = params['path_fmnist_train'] 
        train = datasets.FashionMNIST( root = path_train, train = True, download = True)

    elif(dataset == 4):

        print('\n#--------------------------')
        print('# Loading: Cifar10 Dataset ')
        print('#--------------------------\n')

        name = 'cifar'
        path_train = params['path_cifar_train'] 

        train = datasets.CIFAR10( root = path_train, train = True, download = True)

        train.data = torch.tensor(np.swapaxes(train.data, 1, 3)).type('torch.FloatTensor')
        train.targets = torch.tensor(train.targets)
 
        channel_size = 3

    transforms = params['transforms']
    train.data = data_transforms(train.data, transforms)

    print('Loading -- Complete\n')
    
    return train, channel_size, name


 
