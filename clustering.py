import cv2, numpy as np, random, time, matplotlib.pyplot as plt


class Kcluster:
    def __init__(self, file, clusterNum):
        '''
            Reads a file (either text or image)
            :param file: a filename
            :type file: str

            :param: Represents the number of clusters
            :type clusterNum: int
        '''
        if clusterNum <= 0 or clusterNum > 10:
            print("Number of clusters must be between [0,10]")
            exit()
        try:
            self.data = np.genfromtxt(file)
            self.image = 0
        except:
            self.data = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
            self.image = 1
        self.filename = file
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
        start = time.time()
        center_lookup = dict()
        centers = [np.asarray(mean) for mean in centerDict]
        centerPopulations = [[] for i in range(len(centerDict))]
        if not self.image:
            for point in self.data:
                key = tuple(point)
                if key in center_lookup:
                    nearestCenter = center_lookup[key]
                else:
                    distance = np.linalg.norm(centers[0] - point)
                    for i in range(len(centers)):
                        nextDistance = np.linalg.norm(centers[i] - point)
                        if nextDistance <= distance:
                            distance = nextDistance
                            nearestCenter = i
                    center_lookup[key] = nearestCenter
                centerPopulations[nearestCenter] += [point]

            for populations, center in zip(centerPopulations, centers):
                centerDict[tuple(center)] = populations
            print(f'Time to assign: {time.time() - start} seconds')
            return

        channels = len(self.data[0][0])
        data = np.reshape(self.data, (-1, channels))
        start = time.time()
        for point in data:
            distance = np.linalg.norm(centers[0] - point)
            key = tuple(point)
            if key in center_lookup:
                nearestCenter = center_lookup[key]
            else:
                for i in range(len(centers)):
                    nextDistance = np.linalg.norm(centers[i] - point)
                    if nextDistance <= distance:
                        distance = nextDistance
                        nearestCenter = i
                center_lookup[key] = nearestCenter

            centerPopulations[nearestCenter] += [point]
        for populations, center in zip(centerPopulations, centers):
            centerDict[tuple(center)] = populations
        print(f'Time to assign all points: {time.time() - start} seconds')
        return

    def __computeNewCenters(self):
        '''
            Computes new centers based on the current data for self.centers. Tests if the new set of centers
            retains classification. Returns 1 is convergence is reached. 0 otherwise
        '''
        tempCenters = dict()
        for key in self.centers:
            if not self.centers[key]:
                tempCenters[key] = None
            else:
                tempCenters[tuple(np.mean(self.centers[key], axis=0))] = None
        self.__assignment(tempCenters)

        for key1, key2 in zip(self.centers, tempCenters):
            oCentersList = sorted(self.centers[key1], key=lambda x: x[0])
            tCentersList = sorted(tempCenters[key2], key=lambda x: x[0])
            if not np.array_equal(oCentersList, tCentersList):
                self.centers = tempCenters
                return 0
        return 1

    def __runClustering(self):
        '''
            Runs k-means clustering until convergence between iterations. Returns the list of centers and corresponding points
        '''
        self.__initializeK()
        start = time.time()
        self.__assignment(self.centers)
        while not self.__computeNewCenters():
            continue
        print(f"\tTime to converge runs: {time.time() - start} seconds")
        return self.centers

    def __calculateSquare(self, centers):
        '''
            Calculates the sum of squared errors with respect to distance for each center
            :param centers: dictionary with centers as keys (tuples) and values as numpy arrays
            :type centers: dictionary of numpy arrays
        '''
        SSE = 0
        for key in centers:
            distanceData = np.array([np.sum(np.square((np.subtract(np.asarray(key), point)))) for point in centers[key]])
            averageDistance = np.sum(distanceData)/distanceData.size
            SSE += np.sum(np.square(np.subtract(distanceData, averageDistance)))
        return SSE

    def __cluster_r(self, r):
        '''
            Runs k-means clustering a total of r times and returns the data with the lowest sum of squares error
            :param r: number of runs
            :type r: int
        '''
        self.centers = dict()
        lowestRun = self.__runClustering()
        for runs in range(r - 1):
            newRun = self.__runClustering()
            if self.__calculateSquare(newRun) < self.__calculateSquare(lowestRun):
                lowestRun = newRun
        return lowestRun

    def graph(self, r):
        '''
            Runs the algorithm a total of r times, then creates a graph out of the run with the lowest sum of squares error.
            Also shows an image where all pixels are mapped to the cluster center if an image is specified.
            :param r: number of runs
            :type r: int
        '''
        print("Running k-means cluster algorithm...")
        data = self.__cluster_r(r)
        fig = plt.figure()
        pickAColor = ['aquamarine', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'slategray', 'mediumseagreen']
        if not self.image:
            for key, color in zip(data, pickAColor):
                rotated_data = np.rot90(data[key], k=1)
                x = rotated_data[1]
                y = rotated_data[0]
                plt.scatter(x, y, c=color)
                plt.xlabel('x')
                plt.ylabel('y')
        else:
            ax = fig.add_subplot(projection='3d')
            for key, color in zip(data, pickAColor):
                if len(data[key]) == 0:
                    continue
                rotated_data = np.rot90(data[key], k=1)
                x = rotated_data[2]
                y = rotated_data[1]
                z = rotated_data[0]
                ax.scatter(x, y, z, c=color)
                ax.set_xlabel('Red')
                ax.set_ylabel('Green')
                ax.set_zlabel('Blue')
            pointToPixel = dict()
            for key in data:
                for point in data[key]:
                    if tuple(point) not in pointToPixel:
                        pointToPixel[tuple(point)] = np.asarray(key)
            for line in range(len(self.data)):
                for pixel in range(len(self.data[line])):
                    key = tuple(self.data[line][pixel])
                    self.data[line][pixel] = pointToPixel[key]
            convertedImg = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
            cv2.imshow("mapped pixels to centers", convertedImg)
        
        print(f"SSE: {self.__calculateSquare(data)}")
        plt.title(f'{self.k} clusters from: {self.filename}')
        plt.show()
