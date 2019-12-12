
import os
import yaml
import torch
import argparse
import numpy as np

import eval_model
import train_model

#-------------------------------------
# Initialize: Python Random Seeds All
#-------------------------------------

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#-------------------------------------
# Experiment: Explore Siamese Network
# Supports: Training & Evaluation
#-------------------------------------

def begin_experiment(params):
    
    if(params['train']):
        for current_dim in params['all_dims']:
            params['feature_size'] = current_dim
            train_model.initialize_train(params)
    else:
        eval_model.initialize_eval(params)

#-------------------------------------
# Main: Load YAML Configuration File
#-------------------------------------
 
if __name__ == '__main__':                                                                  
    
    parser = argparse.ArgumentParser()                                                      
    parser.add_argument("-config", help = "Experiment: Train & Eval Siamese Network")              
    args = parser.parse_args()                                                              
                                                                                         
    if(args.config == None):                                                                
        print('\nAttach Configuration File! Run experiment.py -h')                    
        exit()                                                                              
    else:                                                                                   
        print('\nLoading ', args.config,'...') 
    
    params = yaml.load(open(args.config), Loader=yaml.FullLoader) 
   
    begin_experiment(params)
      
