
import numpy as np
import pandas as pd
import skfuzzy as skf
import sklearn.mixture as skm
import sklearn.cluster as skc
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def plot_results(data, orig_labels, result_labels, size, name):

    fig, ax = plt.subplots(1,4)
    fig.suptitle(name+' Clustering Features, Original = '+str(size)) 

    index_a = np.where(orig_labels == 0)
    index_b = np.where(orig_labels == 1)
    class_a = data[index_a]
    class_b = data[index_b]

    ax[0].scatter(x = class_a[:, 0], y = class_a[: , 1], c ='blue')
    ax[1].scatter(x = class_b[:, 0], y = class_b[: , 1], c ='yellow')
    ax[2].scatter(x = data[:, 0], y = data[:, 1], c = orig_labels, cmap = 'plasma')
    ax[3].scatter(x = data[:, 0], y = data[:, 1], c = result_labels, cmap = 'plasma') 

    ax[0].axis('equal') 
    ax[1].axis('equal') 
    ax[2].axis('equal') 
    ax[3].axis('equal') 

    ax[0].set_title('Class A')
    ax[1].set_title('Class B')
    ax[2].set_title('Both A & B')
    ax[3].set_title('Clustering Results')

    plt.show(block=False)
    plt.pause(1)

def gaussian_mixture_model(train, train_labels, test, test_labels, size, plot, plot_dims):

    model = skm.GaussianMixture(n_components=2, covariance_type='full')

    results_train = model.fit_predict(train)
    results_test = model.predict(test)

    if plot:
        plot_results(test, test_labels, results_test, size, "Gaussian Mixture Model")

    return results_train, results_test

def fuzzy_clustering(train, train_labels, test, test_labels, size, plot, plot_dims):

    [center, u, u0, d, jm, p, fpc] = skf.cmeans( train.T, c = 2, m = .5, 
                                                 error = .001, maxiter = 100)

    [nu, nu0, nd, njm, np, nfpc] = skf.cmeans_predict( test.T, center, 3, 
                                                       error = 0.005, maxiter = 1000)
    results_train = u.argmax(axis=0)
    results_test = nu.argmax(axis=0)

    if plot:
        plot_results(test, test_labels, results_test, size, "Fuzzy Clustering")

    return results_train, results_test

def generate_cluster_method(params, plot_dims = [0, 1]):
   
    show = params['show_clusters']
    strategy = params['clustering']
   
    data = params['feature_data']
    test_data = params['test_feature_data']

    size = data['size'] 
    train_labels = data['labels']
    train_samples = data['features']
    
    test_labels = test_data['labels'] 
    test_samples = test_data['features']

    if(len(train_samples.shape) > 2):
        train_samples = np.squeeze(train_samples, axis = 1)
        test_samples = np.squeeze(test_samples, axis = 1) 

    if(strategy == 0):
        r_train, r_test = gaussian_mixture_model( train_samples, train_labels, 
                                                  test_samples, test_labels, size, 
                                                  plot = show, plot_dims = plot_dims )
    elif(strategy == 1):

        r_train, r_test = fuzzy_clustering( train_samples, train_labels, 
                                            test_samples, test_labels, size, 
                                            plot = show, plot_dims = plot_dims )
    elif(strategy == 2):
    
        r_train, r_test = spectral_clustering( train_samples, train_labels, 
                                               test_samples, test_labels, size, 
                                               plot = show, plot_dims = plot_dims )
    else:
        print('\nError: Invalid Clustering Strategy. [0, 1, 2] Supported\n')
        exit()

    return np.asarray(r_train), np.asarray(r_test)
