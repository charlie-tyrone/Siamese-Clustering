
import os
import torch
import pickle
import numpy as np
import siamese_loader as ld                   
import torch.utils.data as tech       

from utils import *
from models import *
from tqdm import tqdm
from clustering import *
from post_analysis import *

#-------------------------------------
# Visualization: Nearest Neighbor(s) 
#-------------------------------------

def display_KNN(sample, anchors, distances, pred, truth):
    
    fig, ax = plt.subplots(1,3)
    fig.suptitle('Model Predicts Label: '+str(pred)+', Truth: '+str(truth)) 

    sample = torch.squeeze(sample).detach().cpu().numpy()
    anchor_A = torch.squeeze(anchors[0]).detach().cpu().numpy()
    anchor_B = torch.squeeze(anchors[1]).detach().cpu().numpy()

    ax[0].imshow(sample)
    ax[1].imshow(anchor_A)
    ax[2].imshow(anchor_B)

    ax[0].set_title('Test Sample')
    ax[1].set_title('Distance: ' +str(distances[0]))
    ax[2].set_title('Distance: ' +str(distances[1]))

    plt.show(block=False)
    input()


#-------------------------------------
# Algorithm: K-Nearest Neighboor 
# Strategy: K > 1, Majority Wins 
#-------------------------------------

def KNN(distances, anchor_labels):

    possible_classes = np.unique(anchor_labels)

    if( len(possible_classes) > 1):
        num_votes = np.zeros(len(possible_classes))
        for count, current in enumerate(possible_classes):
            for match in anchor_labels:
                if(current == match):
                    num_votes[count] = num_votes[count] + 1
        prediction = np.argmax(num_votes)
    else:
        prediction = possible_classes[0]
  
    return prediction 

#-------------------------------------
# Test: Siamese Neural Network (SNN)
#-------------------------------------

def test_epoch(params):

    #---------------------------------
    # Load: Testing Parameters 
    # 1) Data: Testing Dataset
    # 2) Model: Siamese Network
    # 3,4) KNN General Parameters
    #---------------------------------

    data = params['eval']                                                                   
    model = params['model']                                                                 
    k_value = params['k_value']                                                             
    show_test = params['show_test']

    gpu_flag = params['gpu']
    device = torch.device('cuda:'+str(params['device']))

    #---------------------------------
    # Evaluation: KNN, Input Features
    # 1) Inputs --> SNN --> Distances
    # 2) Distances --> KNN --> Output
    # 3) Scoring Basic Precision
    #---------------------------------
    
    model.eval()
    
    accuracy = []
    for batch_id, test_params in enumerate(tqdm(data, desc='Test')):
 
        #-----------------------------
        # Dataset: Siamese Testing
        # 1) Anchors (Prototypes)
        # 2) Anchor Labels (Class)
        # 3,4) Test Sample & Label 
        #-----------------------------

        anchors, anchor_labels, sample, label = test_params 
        label = label.cpu().numpy()

        anchors = torch.stack(anchors) 
        anchors = torch.squeeze(anchors, dim=1) 
        anchor_labels = torch.squeeze(anchor_labels, dim=0).cpu().numpy()
         
        num_samples = anchors.shape[0]
        copy_samples = torch.zeros( num_samples, sample.shape[1], 
                                    sample.shape[2], sample.shape[3])
        copy_samples[:] = sample

        if(gpu_flag): 
            anchors = anchors.type('torch.FloatTensor').to(device)
            copy_samples = copy_samples.type('torch.FloatTensor').to(device)

        #-----------------------------
        # Evaluation: Siamese Model
        # Inputs --> SNN --> Distance
        # Distance == Dissimilarity
        #-----------------------------

        distances, _ = model(anchors, copy_samples)
        orig_distances = distances.detach().cpu().numpy()
 
        #-----------------------------
        # Prediction: KNN Algorithm
        # Find Most Similar Anchor(s)
        #-----------------------------

        indices = np.argsort(orig_distances)
        distances = distances[indices][:k_value] 
        anchor_labels = anchor_labels[indices][:k_value]

        prediction = KNN(distances, anchor_labels) 
        
        if(show_test):
            anchors = anchors[indices]
            orig_distances = orig_distances[indices] 
            display_KNN(sample, anchors, orig_distances, prediction, label)

        accuracy.append(1) if(prediction == label) else accuracy.append(0)

    return sum(accuracy) / len(accuracy)       


def load_models(params):

    all_sizes = params['all_dims']
    input_size = params['input_size']
    best_models = params['best_models']
    
    gpu_flag = params['gpu']
    use_all_gpu = params['gpu_all']
    device = torch.device('cuda:'+str(params['device']))
   
    all_models = []
    for current_path, current_size in zip(best_models, all_sizes):
        model = Siamese_Model(input_size, current_size)
        model.load_state_dict(torch.load(current_path))
        
        if(gpu_flag): 
            if(use_all_gpu):
                model = torch.nn.DataParallel(model)
            else:    
                model = model.to(device)

        all_models.append(model)


    return all_models

#----------------------------i------------
# Initialize: Dataloaders, Test Layout
#-----------------------------------------

def initialize_eval(params):

    #-------------------------------------
    # Load: Testing Dataset, DataLoader
    #-------------------------------------
 
    if(params['get_train_features']):
    
        data, params['input_size'], params['name'] = load_train_dataset(params)
        params['data'] = data
        data = ld.Siamese_Dataset(params, 'test')
        params['feats'] = tech.DataLoader(dataset = data, shuffle = False, batch_size = 1)

    _, _, test_dataset, params['input_size'], params['name'] = load_dataset(params)

    #-------------------------------------
    # Load: Trained Siamese Networks
    #-------------------------------------
    
    all_sizes = params['all_dims']
    best_models, best_results = find_best_models(params)
   
    print('\nGathering Best Trained Models\n')
 
    for i, (model, result, size) in enumerate(zip(best_models, best_results, all_sizes)):

        result = np.round(result, 4)
        print('Features: ', size, ', Model Path:', model,', Acc:', result)

    params['best_models'] = best_models
    params['best_models'] = load_models(params)

    print('\nComplete - Best Models Loaded\n')
    input('Press [Enter] To Continue....')

    #-------------------------------------
    # Optional: Generate Train Features
    #-------------------------------------

    if(params['get_train_features']):
    
        print('\nOperation: SNN Feature Generation (I/O)\n')

        generate_features(params)
        
        print('\nComplete - SNN Feature Generation \n')
     
    #-------------------------------------
    # Load: Data Features (All Models)
    #-------------------------------------
   
    print('Operation: Loading Dataset Features\n')
    
    all_feature_data = load_features(params)

    print('\nComplete - All Features Loaded \n')
   
    #-------------------------------------
    # Optional: Dimensionality Reduction
    #-------------------------------------
    
    if(params['reduce_size']):

        print('Operation: Feature Reduction\n')

        params['feature_data'] = all_feature_data

        all_feature_data = reduce_features(params)

        print('\nComplete - Feature Reduction\n')

    #-------------------------------------
    # Clustering: Feature Space Data
    # 1) Generate New Anchors, Test Model
    # 2) Test Model, Clustering vs KNN
    #-------------------------------------

    final_results = []
    if(params['new_anchors']):
        for current_model, current_data in zip(params['best_models'], all_feature_data): 
            params['feature_data'] = current_data
            params['test_feature_data'] = params['feature_data']

            results_train, results_test = generate_cluster_method(params)        
            current_data['cluster_results'] = results_test

            #-----------------------------
            # Initialize: New Anchors
            #-----------------------------
            
            #params['path_anchors'] = save_new_anchors(params) 
            _, _, test_dataset, params['input_size'], params['name'] = load_dataset(params)
            params['data'] = test_dataset
            test = ld.Siamese_Dataset(params, 'test')
            params['eval'] = tech.DataLoader( dataset = test, 
                                              shuffle = True, batch_size = 1)                 
           
            print(params['path_anchors'])

            #-----------------------------
            # Evaluate: New Anchors (SNN)
            #-----------------------------

            params['model'] = current_model

            final_results.append(test_epoch(params)) 

    else:

        for count, (current_model, current_data) in enumerate(zip( params['best_models'], 
                                                                   all_feature_data )): 

            _, _, test_dataset, params['input_size'], params['name'] = load_dataset(params)
            params['data'] = test_dataset
            test = ld.Siamese_Dataset(params, 'test')
            params['feat'] = tech.DataLoader( dataset = test, 
                                              shuffle = True, batch_size = 1)                 

            print('Operation: Loading Test Dataset Features\n')
            
            test_feature_data = load_features(params)

            print('\nComplete - All Test Features Loaded \n')

            params['feature_data'] = current_data
            params['test_feature_data'] = test_feature_data[count]

            _, params['results_test'] = generate_cluster_method(params)        

            final_results.append(calculate_accuracy(params))  

    print('\nFinal Results\n')

    for size, acc in zip(all_sizes, final_results):
        print('Model:',size,', Accuracy:', np.round(acc, 4)) 

