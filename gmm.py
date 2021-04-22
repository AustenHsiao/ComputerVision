from clustering import *
import numpy as np


class Gmm(Kcluster):
    def __init__(self, file, cluster):
        super().__init__(file, cluster)
        self.data = super().runClustering()
        # print(self.__computeCentroid())
        self.N = sum([len(self.data[key]) for key in self.data])

    def __computeInitCentroid(self):
        '''
        Updates self.data with cluster centroids. Should be used for the initialization only
        '''
        for key in self.data:
            newCenter = tuple(np.mean(self.data[key], axis=0))
            if newCenter != key:
                self.data[newCenter] = self.data[key]
                del self.data[key]
                continue

    def __computeInitCovariance(self, cluster):
        '''
        :param cluster: returns the covariance matrix associated with the cluster of data points
        :type cluster: np array of np arrays
        '''
        return np.cov(cluster)

    def __calculateInitPrior(self, cluster):
        '''
        :param cluster: returns the estimated prior for the given cluster 
        :type cluster: np array of np arrays
        '''
        return len(cluster)/self.N

    def __calculatePosterior(self):
