
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from dim_reduction import *

def calculate_accuracy(params):

    test_data = params['test_feature_data']
    test_samples = test_data['features']
    test_labels = test_data['labels'] 

    predictions = params['results_test'];

    accuracy = []

    for current_truth, current_pred in zip(test_labels, predictions):
        
        if(current_truth == current_pred):
            accuracy.append(1)

        else:
            accuracy.append(0)

    accuracy = sum(accuracy) / len(accuracy)  

    #print('Final Accuracy:', accuracy)

    return accuracy


def save_new_anchors(params):

    name = params['name']
    data = params['feature_data']
    path_new_anchors = params['path_new_anchors']

    size = data['size'] 
    labels = data['labels']
    features = data['features']
    orig_samples = data['orig_samples']

    cluster_labels = data['cluster_results']
    path_new_anchors = os.path.join(path_new_anchors, name, str(size))

    unique_labels = np.unique(cluster_labels)

    for current_label in unique_labels:
        path_save = os.path.join(path_new_anchors, str(current_label))
        initialize_folder(path_save)
   
        index = np.where(cluster_labels == current_label)
        class_samples = orig_samples[index]

        sample_idx = np.random.randint(class_samples.shape[0])
        new_anchor = class_samples[sample_idx]
        
        if(len(new_anchor.shape) == 3):
            new_anchor = np.swapaxes(new_anchor, 0, 2)

        plt.imsave(os.path.join(path_save, str(current_label)+'.png'), new_anchor)

    return path_new_anchors

def reduce_features(params):

    new_feature_data = [] 
    show_plots = params['show_reduce']
    all_feature_data = params['feature_data']

    for count, current_data in enumerate(all_feature_data):
 
        new_data = {}
        
        orig_samples = current_data['orig_samples']
        samples = current_data['features']
        labels = current_data['labels']
        size = current_data['size']

        if(show_plots):
            fig, ax = plt.subplots(1, 3, figsize=(6,6))
            fig.suptitle('Class Features, Original = '+str(size)) 
        
        if(samples.shape[-1] > 2):
            samples = np.squeeze(samples, axis = 1)
            samples = reduce_dims(samples, params['reduction'])

        samples = samples.reshape(samples.shape[0], 2)

        new_data['orig_samples'] = orig_samples
        new_data['features'] = samples
        new_data['labels'] = labels
        new_data['size'] = size

        new_feature_data.append(new_data)

        if(show_plots):
            a = np.where(labels == 0)
            b = np.where(labels == 1)

            c0 = samples[a]
            l0 = labels[a]

            c1 = samples[b]
            l1 = labels[b]

            ax[0].scatter(x=c0[:, 0], y=c0[:, 1], c='blue')
            ax[1].scatter(x=c1[:, 0], y=c1[:, 1], c='yellow')
            ax[2].scatter(x=samples[:, 0], y=samples[:, 1], c=labels, cmap='plasma')

            ax[0].axis('equal') 
            ax[1].axis('equal') 
            ax[2].axis('equal') 

            ax[0].set_title('Class A')
            ax[1].set_title('Class B')
            ax[2].set_title('Both A & B')

            #plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)

    return new_feature_data
     
def load_features(params):

    path_features = os.path.join(params['path_train_features'], params['name'])

    all_data_features = np.asarray(os.listdir(path_features))
    all_data_features = [int(ele) for ele in all_data_features]
    all_data_features.sort()


    results = []
    for current_feature in all_data_features:

        all_details = {}
        all_features, all_samples, all_labels = [], [], []
        path = os.path.join(path_features, str(current_feature))
        data_file = os.path.join(path, os.listdir(path)[0])

        with open(data_file, 'rb') as handle:
            data = pickle.load(handle)
            all_keys = data.keys()
            
            for current_key in tqdm(all_keys,desc=str(current_feature)):
                features, label, sample = data[current_key]
                all_features.append(features)
                all_samples.append(sample)
                all_labels.append(label[0])
                
        all_details['size'] = current_feature
        all_details['labels'] = np.asarray(all_labels)
        all_details['features'] = np.asarray(all_features)
        all_details['orig_samples'] = np.asarray(all_samples)

        results.append(all_details)

    return results

def generate_features(params):

    data = params['feats'] 
    all_dims = params['all_dims']                                                                
    all_models = params['best_models']

    #---------------------------------
    # Load: General GPU Parameters
    #---------------------------------

    gpu_flag = params['gpu']
    use_all_gpu = params['gpu_all']
    device = torch.device('cuda:'+str(params['device']))

    #---------------------------------
    # Save: Train Features, All Models
    #---------------------------------

    print('Gathering Dataset Features\n')

    for count, (model, feature_size) in enumerate(zip(all_models, all_dims)):
        
        all_features = {}       
        name = params['name'] 
        path_features = params['path_train_features']
        path_features = os.path.join(path_features, name, str(feature_size))
        initialize_folder(path_features)

        if(gpu_flag): 
            if(use_all_gpu):
                model = torch.nn.DataParallel(model)
            else:    
                model = model.to(device)

        model.eval()                                                                       
        
        for batch_id, test_params in enumerate(tqdm(data, desc=str(feature_size))):
     
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
             
            if(gpu_flag): 
                anchors = anchors.type('torch.FloatTensor').to(device)
                sample = sample.type('torch.FloatTensor').to(device)

            #-----------------------------
            # Evaluation: Siamese Model
            # Inputs --> SNN --> Distance
            # Distance == Dissimilarity
            #-----------------------------

            _, all_embeddings = model(sample, sample)

            sample_features = all_embeddings['nd_input_2']

            sample_features = sample_features.detach().cpu().numpy()
            sample = torch.squeeze(sample).detach().cpu().numpy()
            
            all_features[str(batch_id)] = [sample_features, label, sample]
            
        path_save = os.path.join(path_features, 'results.p')
        with open(path_save, 'wb') as handle:
            pickle.dump(all_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nComplete - All Features Saved \n')

def find_best_models(params):

    num_epochs = params['num_epochs']
    path_models = os.path.join(params['path_models'], params['name'])
    path_results = os.path.join(params['path_results'], params['name'])

    best_models = []
    best_results = []
    all_features_models = np.asarray(os.listdir(path_models))
    all_features_scores = np.asarray(os.listdir(path_results)) 
    
    all_features_scores = [int(ele) for ele in all_features_scores]
    all_features_models = [int(ele) for ele in all_features_models]

    all_features_models.sort()
    all_features_scores.sort()

    all_features_scores = [str(ele) for ele in all_features_scores]
    all_features_models = [str(ele) for ele in all_features_models]

    for models_feature, scores_feature in zip(all_features_models, all_features_scores):

        scores_feature = os.path.join(path_results, scores_feature)
        models_feature = os.path.join(path_models, models_feature)

        data_file = os.path.join(scores_feature, os.listdir(scores_feature)[0])
        current_models = os.listdir(models_feature)

        with open(data_file, 'rb') as handle:
            params = pickle.load(handle)
    
            train = params['train']
            valid = params['valid']
            test = params['test']
            
            model_acc = np.max(test)
            model_index = np.argmax(test)
        
        best_models.append(os.path.join(models_feature, current_models[model_index]))
        best_results.append(model_acc) 

    return best_models, best_results 


