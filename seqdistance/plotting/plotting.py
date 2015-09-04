"""
Functions for basic visualization and clustering of pairwise-distance matrices.
"""
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA,KernelPCA
from sklearn import cluster
from sklearn.manifold import Isomap

try:
    import tsne
    #import pytsne
except ImportError:
    print "seqdistance: Could not load tsne: will be unavailable for embedding."

"""TODO:
    (1) Wrap imports in a try so that important doesn't fail without tsne
    (2) Add the plotting function for visualizing an embedding."""

def embedDistanceMatrix(dist, method='tsne'):
    """MDS embedding of sequence distances in dist, returning Nx2 x,y-coords: tsne, isomap, pca, mds, kpca"""
    if method == 'tsne':
        xy = tsne.run_tsne(dist,no_dims=2)
        #xy=pytsne.run_tsne(adist,no_dims=2)
    elif method == 'isomap':
        isoObj = Isomap(n_neighbors=10,n_components=2)
        xy = isoObj.fit_transform(dist)
    elif method == 'mds':
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=15,
                           dissimilarity="precomputed", n_jobs=1)
        xy = mds.fit(dist).embedding_
        rot = PCA(n_components=2)
        xy = rot.fit_transform(xy)
    elif method == 'pca':
        pcaObj = PCA(n_components=2)
        xy = pcaObj.fit_transform(1-dist)
    elif method == 'kpca':
        pcaObj = KernelPCA(n_components=2,kernel='precomputed')
        xy = pcaObj.fit_transform(1-dist)
    elif method == 'lle':
        lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2,method='standard')
        xy = lle.fit_transform(dist)
    return xy