
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def reduce_dims(data, strategy):

    current_size = data.shape[-1]
   
    if(strategy == 0):
        print('\nCurrent Feature Size:', current_size, ', Invoking PCA --> 2\n')
        pca = PCA(n_components = 2)
        results = pca.fit_transform(data)
        ratio = pca.explained_variance_ratio_
        print('Variation, Each Principal Component: {}'.format(ratio))


    if(strategy == 1):
 
        print('\nCurrent Feature Size:', current_size, ', Invoking TSNE --> 2\n')
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        results = tsne.fit_transform(data) 

    if(strategy == 2):

        if(current_size < 20):
            strategy = 1
            results = reduce_dims(data, strategy)
        else:
            print('\nCurrent Feature Size:', current_size, 
                  ', Invoking PCA --> 20 && TSNE --> 2\n')

            pca = PCA(n_components = 20)
            data = pca.fit_transform(data)
            ratio = pca.explained_variance_ratio_
            print('Variation, Each Principal Component: {}'.format(ratio))

            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            results = tsne.fit_transform(data) 

    return results
