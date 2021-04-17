import cv2
import numpy as np
import random


class Kcluster:
    def __init__(self, file, clusterNum):
        '''
            Reads a file (either text or image)
            :param file: a filename
            :type file: str

            :param: Represents the number of clusters
            :type clusterNum: int
        '''
        try:
            self.data = np.genfromtxt(file)
            self.image = 0
        except:
            self.data = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
            self.image = 1

        self.k = clusterNum
        self.centers = dict()

    def __initializeK(self):
        '''
            No parameters. Populates the keys for dictionary 'self.centers'. This will hang if you choose too many clusters or you do not have many distinct data points
        '''

        if not self.image:
            while len(self.centers) != self.k:
                self.centers[tuple(self.data[np.random.randint(0, len(self.data))])] = None
            return
        channels = len(self.data[0][0])
        data = np.reshape(self.data, (-1, channels))
        while len(self.centers) != self.k:
            self.centers[tuple(data[np.random.randint(0, len(self.data))])] = None

    def __assignment(self, centerDict):
        '''
            populates centerDict{} with points based on proximity
        '''
        centers = [np.asarray(mean) for mean in centerDict]
        if not self.image:
            for point in self.data:
                nearest = np.argmin(np.array(
                    [np.sqrt(np.sum(np.square(np.subtract(point, mean)))) for mean in centers]))
                correspondingCenter = tuple(centers[nearest])
                if np.any(centerDict[correspondingCenter]) == None:
                    centerDict[correspondingCenter] = np.array([point])
                else:
                    centerDict[correspondingCenter] = np.append(
                        centerDict[correspondingCenter], [point], axis=0)
            return
        channels = len(self.data[0][0])
        data = np.reshape(self.data, (-1, channels))
        for point in data:
            nearest = np.argmin(np.array([np.sqrt(np.sum(np.square(np.subtract(point, mean)))) for mean in centers]))
            correspondingCenter = tuple(centers[nearest])
            if np.any(centerDict[correspondingCenter]) == None:
                centerDict[correspondingCenter] = np.array([point])
            else:
                centerDict[correspondingCenter] = np.append(
                    centerDict[correspondingCenter], [point], axis=0)
        return

    def __computeNewCenters(self):
        '''
            Computes new centers based on the current data for self.centers. Tests if the new set of centers
            retains classification. Returns 1 is convergence is reached. 0 otherwise
        '''
        tempCenters = dict()
        for key in self.centers:
            tempCenters[tuple(np.mean(self.centers[key], axis=0))] = None

        self.__assignment(tempCenters)

        for key1, key2 in zip(self.centers, tempCenters):
            if not np.array_equal(self.centers[key1], tempCenters[key2]):
                self.centers = tempCenters
                return 0
        return 1

    def __runClustering(self):
        '''
            Runs k-means clustering until convergence between iterations. Returns the list of centers and corresponding points
        '''
        self.__initializeK()
        self.__assignment(self.centers)
        while not self.__computeNewCenters():
            continue
        return self.centers

    def __calculateSquare(self, centers):
        '''
            Calculates the sum of squared errors with respect to distance for each center
        '''
        SSE = 0
        for key in centers:
            distanceData = np.array([np.sum(np.square((np.subtract(np.asarray(key), point)))) for point in centers[key]])
            averageDistance = np.sum(distanceData)/distanceData.size
            SSE += np.sum(np.square(np.subtract(distanceData, averageDistance)))
        print(SSE)
        return SSE

    def cluster_r(self, r):
        '''
            Runs k-means clustering a total of r times and returns the data with the lowest sum of squares error
        '''
        self.centers = dict()
        lowestRun = self.__runClustering()
        for runs in range(r - 1):
            newRun = self.__runClustering()
            if self.__calculateSquare(newRun) < self.__calculateSquare(lowestRun):
                lowestRun = newRun
        return lowestRun
    
    
