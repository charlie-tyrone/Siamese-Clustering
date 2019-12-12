
import os                             
import torch                          
import pickle                         
import subprocess                     
import numpy as np                    
import siamese_loader as ld                   
import torch.utils.data as tech       


from utils import *
from models import *
from tqdm import tqdm                 
from eval_model import *

#-------------------------------------
# Train || Validate: Siamese Network
#-------------------------------------

def train_valid_epoch(params, name):

    #---------------------------------
    # Load: Train / Valid Parameters 
    # 1) Mode: Train || Valid 
    # 2) Data: Dataset, Train || Valid
    # 3) Model: Siamese Neural Network
    #---------------------------------

    mode = params['mode']                                                                  
    data = params['data']                                                                  
    model = params['model']                                                                
 
    gpu_flag = params['gpu']
    device = torch.device('cuda:'+str(params['device']))

    if(mode.lower() == 'train'): 
        optimizer = params['optimizer'] 
        model.train()                                                                      
    else:
        model.eval()                                                                       

    #---------------------------------
    # Evaluation: Train || Validation
    #---------------------------------

    epoch_loss = []                                                                        
    valid_count = 0

    for batch_id, train_params in enumerate(tqdm(data, desc=name)):                                   

        #-----------------------------
        # Dataset: Siamese Train/Valid 
        # 1,2) Sample Pairwise Inputs 
        # 3) Label (Same || Different) 
        #-----------------------------

        images_1, images_2, labels = train_params 
        
        if(gpu_flag): 
            images_1 = images_1.type('torch.FloatTensor').to(device)                   
            images_2 = images_2.type('torch.FloatTensor').to(device)                   
            labels = labels.type('torch.FloatTensor').to(device)                       

        #-----------------------------
        # Evaluation: Siamese Model
        # Inputs --> SNN --> Distance
        # Distance == Dissimilarity
        #-----------------------------

        inputs = [ images_1, images_2 ]
        distances, _ = model(images_1, images_2)

        #-----------------------------
        # Calculate: Siamese Loss
        # If Train: Backpropagation 
        #-----------------------------

        loss = calc_siamese_loss(distances, labels)  
        epoch_loss.append(loss.item())

        if(mode.lower() == 'train'):
            optimizer.zero_grad()                                                      
            loss.backward()                                                            
            optimizer.step()                                                           
    
    epoch_loss = sum(epoch_loss)/len(epoch_loss)                                       

    if(mode.lower() == 'train'):
        return model, epoch_loss
    else:
        return epoch_loss

#----------------------------i------------
# Train: Convolutional Nerual Network 
#-----------------------------------------

def train_new_model(params):

    #---------------------------------
    # Load: Data, Train & Valid & Test 
    #---------------------------------

    test = params['test']                                                                   
    train = params['train']                                                                 
    valid = params['valid']                                                                 

    #---------------------------------
    # Load: General Data Parameters
    #---------------------------------

    input_size = params['input_size']
    feature_size = params['feature_size']

    #---------------------------------
    # Load: Neural Network Parameters
    #---------------------------------

    optim = params['optimizer']                                                             
    num_epochs = params['num_epochs']                                                       
    momentum_rate = params['momentum_rate']                                                 
    learning_rate = params['learning_rate']                                                 

    #---------------------------------
    # Load: General GPU Parameters
    #---------------------------------

    gpu_flag = params['gpu']
    use_all_gpu = params['gpu_all']
    device = torch.device('cuda:'+str(params['device']))
   
    #---------------------------------
    # Load: Paths, All Train Results
    #---------------------------------
    
    name = params['name']
    path_models = params['path_models']
    path_results = params['path_results']
    
    path_models = os.path.join(path_models, name, str(feature_size))
    path_results = os.path.join(path_results, name, str(feature_size))

    initialize_folder(path_models)
    initialize_folder(path_results)

    print('\n#----------------------------')
    print('# Initialize: Siamese Network')
    print('#----------------------------\n')

    model = Siamese_Model(input_size, feature_size) 
 
    #---------------------------------
    # GPU Options: 1 GPU || All GPUs
    #---------------------------------
   
    if(gpu_flag): 
        if(use_all_gpu):
            model = torch.nn.DataParallel(model)
        else:    
            model = model.to(device)

    #---------------------------------
    # Optimizer Options: SGD || ADAM
    #---------------------------------

    if(optim == 1):
        optimizer = torch.optim.SGD( model.parameters(), lr = learning_rate, 
                                     momentum = momentum_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    params['optimizer'] = optimizer
 
    print('Complete - Loaded Model')

    print('\n#----------------------------')
    print('# Training: Siamese Network ')
    print('#----------------------------\n')
 
    test_accuracy = []                                                                      
    valid_epoch_loss = []                                                                   
    train_epoch_loss = []                                                                   

    for epoch in range(num_epochs):                                                         
       
        print('Current Epoch:', epoch+1,'\n') 
 
        #-----------------------------
        # Training: Siamese Model
        # Scoring: Training Loss
        #-----------------------------

        params['model'] = model
        params['data'] = train
        params['mode'] = 'train' 
        model, epoch_loss  = train_valid_epoch(params, 'Train') 
        train_epoch_loss.append(epoch_loss)    
 
        #-----------------------------
        # Validation: Siamese Model
        # Scoring: Validation Loss 
        #-----------------------------
 
        params['data'] = valid
        params['mode'] = 'valid' 
        epoch_loss = train_valid_epoch(params, 'Valid') 
        valid_epoch_loss.append(epoch_loss)
        
        #-----------------------------
        # Testing: Siamese Model
        # Scoring: Model Precision
        #-----------------------------

        params['data'] = test
        accuracy = test_epoch(params)
        test_accuracy.append(accuracy)

        print('\n#------------------------')
        print('# Epoch Results:',epoch + 1)
        print('#------------------------\n')

        print( 'Epoch: ', epoch + 1, ', Num-Features: ', feature_size,
               '\nTrain-Loss Avergae: ', train_epoch_loss[epoch],
               '\nValid-Loss Average: ', valid_epoch_loss[epoch], 
               '\nTesting Accuracy: ', test_accuracy[epoch] )

        print('\n#------------------------\n')

        #-----------------------------
        # Save: Current Simaese Model 
        #-----------------------------

        path_save = os.path.join(path_models, 'epoch_'+str(epoch)+'.pt')
        torch.save(model.state_dict(), path_save)

    print('\n#----------------------------')
    print('# Training Done: Saving Results')
    print('#----------------------------\n')

    #---------------------------------
    # Save: Loss && Precision Plots
    #---------------------------------

    plot_results = {}
    plot_results['train'] = train_epoch_loss
    plot_results['valid'] = valid_epoch_loss
    plot_results['test'] = test_accuracy

    path_save = os.path.join(path_results, 'results.p')
    with open(path_save, 'wb') as handle:
        pickle.dump(plot_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Complete - Results Saved, Features:', feature_size, '\n')


#----------------------------i------------
# Initialize: Dataloaders, Train Layout
#-----------------------------------------

def initialize_train(params):
    
    show_train = params['show_train'] 
    batch_size = params['batch_size'] 

    #-------------------------------------
    # Load: Siamese Datasets, DataLoaders
    #-------------------------------------
 
    train, valid, test, size, name = load_dataset(params)

    params['name'] = name
    params['input_size'] = size   
 
    params['data'] = train
    train = ld.Siamese_Dataset(params, 'train')                                             
    print()

    params['data'] = valid
    valid = ld.Siamese_Dataset(params, 'valid')                                             
    print()

    params['eval'] = test
    test = ld.Siamese_Dataset(params, 'test')

    train = tech.DataLoader(dataset = train, shuffle = True, batch_size = batch_size)       
    valid = tech.DataLoader(dataset = valid, shuffle = False, batch_size = batch_size)      
    test = tech.DataLoader(dataset = test, shuffle = False, batch_size = 1)                 
 
    print('\nComplete - Loaded Augmented Dataset')

    if(show_train):
        plot_train(train)
    
    #-------------------------------------
    # Finalize: Datasets, Begin Pipeline
    #-------------------------------------

    params['train'] = train
    params['valid'] = valid
    params['test'] = test
    
    train_new_model(params)

