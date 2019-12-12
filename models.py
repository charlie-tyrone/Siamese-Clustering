
import torch                                                                               
import numpy as np
import torch.nn as nn                                                                      
import torch.nn.functional as F

#-------------------------------------
# Initialize: Siamese Neural Network
#-------------------------------------

class Siamese_Model(nn.Module):
    
    def __init__(self, size, num_features):

        super(Siamese_Model, self).__init__()

        #-----------------------------
        # Model: Feature Extraction 
        #-----------------------------

        self.extract = nn.Sequential( nn.Conv2d( in_channels = size, out_channels = 32, 
                                                 kernel_size = 5, stride = 1, padding = 2),
                                                 nn.BatchNorm2d(32),
                                                 nn.ReLU(inplace = True),

                                      nn.Conv2d( in_channels = 32, out_channels = 64, 
                                                 kernel_size = 5, stride = 2, padding = 2),
                                                 nn.BatchNorm2d(64),
                                                 nn.ReLU(inplace = True),

                                      nn.Conv2d( in_channels = 64, out_channels = 64, 
                                                 kernel_size = 5, stride = 2, padding = 2),
                                                 nn.BatchNorm2d(64),
                                                 nn.ReLU(inplace = True) )

        #-----------------------------
        # Model: Dimension Reduction  
        #-----------------------------

        self.decimate = nn.Sequential( nn.Linear(64 * 7 * 7, 512),
                                       nn.ReLU(inplace = True),
                                       nn.Linear(512, num_features) )
 
        #-----------------------------
        # Model: Force 2D Embeddings 
        #-----------------------------

        self.embeddings = nn.Sequential( nn.Linear(64* 7 * 7, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, 2) )

    #-------------------------------
    # Forward Pass: Siamese Network
    #-------------------------------

    def forward(self, x1, x2):

        #-----------------------------
        # Get: All Features, Input 1 
        #-----------------------------

        features_x1 = self.extract(x1)
        features_x1_b = features_x1.view(features_x1.size()[0], -1)
        features_x1_s = self.decimate(features_x1_b)        
 
        #-----------------------------
        # Get: All Features, Input 2 
        #-----------------------------

        features_x2 = self.extract(x2)
        features_x2_b = features_x2.view(features_x2.size()[0], -1)
        features_x2_s = self.decimate(features_x2_b)

        #-----------------------------
        # Calculate: Dissimilarity
        # Get: N-D, 2-D Embeddings 
        #-----------------------------

        distance = F.pairwise_distance(features_x1_s, features_x2_s)

        all_embeddings = {}
        all_embeddings['nd_input_1'] = features_x1_s 
        all_embeddings['nd_input_2'] = features_x2_s
        all_embeddings['2d_input_1'] = self.embeddings(features_x1_b) 
        all_embeddings['2d_input_2'] = self.embeddings(features_x2_b) 

        return distance, all_embeddings

def calc_siamese_loss(pred_sim, truth_sim, choice=0):

    if(choice == 1):
        
        # Cross Entropy Loss
        sigmoid = nn.Sigmoid()
        pred_sim = sigmoid(pred_sim)
        siamese_loss = F.binary_cross_entropy(pred_sim, truth_sim)  
    
    else:
        
        # Contrastive Loss
        m = 1
        y = truth_sim
        dw = pred_sim
        loss = ( (1 - y) * 1/2 * torch.pow(dw, 2) + y * 1/2 * torch.pow(torch.clamp(m - dw, min = 0.0), 2) ) 
        siamese_loss = torch.mean(loss)

    return siamese_loss

