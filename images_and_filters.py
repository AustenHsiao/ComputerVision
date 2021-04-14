''' 
Computer vision and deep learning. Written by Austen Hsiao
'''

'''
Part 1: Read in "filter1_img.jpg" and "filter2_img.jpg", then apply 3x3 and 5x5 Gaussian filters using 
'''




import random
import cv2
import numpy as np
class applyGFilter:
    def __init__(self, file):
        try:
            self.image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        except:
            print("Error reading file.")
            exit()
        self.x = len(self.image[0])
        self.y = len(self.image)

    def showcase(self):
        '''
            Displays image results for all filters
        '''
        cv2.imshow("originalImage", self.image)
        cv2.imshow("3x3 Gauss Filtered", self.apply3Gauss())
        cv2.imshow("5x5 Gauss Filtered", self.apply5Gauss())
        cv2.imshow(f"Derivative of Gauss (x) Filtered", self.applyDoG('x'))
        cv2.imshow(f"Derivative of Gauss (y) Filtered", self.applyDoG('y'))
        cv2.imshow(f"Sobel Filtered", self.applySobel())
        cv2.waitKey(0)

    def apply3Gauss(self):
        '''
            No parameters. Returns an np array of the supplied image after convolution with a 3x3 Gauss filter
        '''
        mask = np.divide(np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]), 16)

        # pad 0s, width 1
        image = np.pad(self.image, 1, mode='constant', constant_values=0)

        # convolution
        return self.convolute(mask, image)

    def apply5Gauss(self):
        '''
            No parameters. Returns an np array of the supplied image after convolution with a 5x5 Gauss filter
        '''
        mask = np.divide(np.array([[1, 4, 7, 4, 1],
                                   [4, 16, 26, 16, 4],
                                   [7, 26, 41, 26, 7],
                                   [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]]), 273)

        # pad 0s, width 2
        image = np.pad(self.image, 2, mode='constant', constant_values=0)

        # convolution
        return self.convolute(mask, image)

    def applyDoG(self, direction):
        '''
            Returns an np array of the supplied image after convolution with a derivative of 
            Gaussian filter, specified with 'x' or 'y'
            :param direction: specifies the direction of the DoG filter
            :type direction: char
        '''
        if direction.lower() == 'x':
            mask = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]])
        elif direction.lower() == 'y':
            mask = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]])
        else:
            print(
                "The direction for the DoG filter can only be 'x' or 'y'.\n(eg. applyDoG('x'))")
            exit()

        # pad 0s, width 1
        image = np.pad(self.image, 1, mode='constant', constant_values=0)

        # convolution
        return self.convolute(mask, image)

    def applySobel(self):
        '''
            No parameters. Returns an np array of the supplied image after convolution with the derivative of Gauss
            filters to produce the Sobel filtered result
        '''
        return np.sqrt(np.add(np.square(self.applyDoG('x')), np.square(self.applyDoG('y'))))

    def convolute(self, mask, image):
        '''
        convolutes an image matrix with the given mask. 
        :param mask: filter matrix
        :type mask: numpy array

        :param image: image matrix (single channel)
        :type image: numpy array
        '''
        maskDimension = len(mask)
        xDimension = len(image[0])
        yDimension = len(image)
        return np.divide(np.array([[np.dot(image[y:(y + maskDimension), x:(x + maskDimension)].flatten(), mask.flatten()) for x in range(0, xDimension - (maskDimension - 1))] for y in range(0, yDimension - (maskDimension - 1))]), 256)


'''
Part 2: Read in some files and perform k-means clustering 
'''


class KMeansCluster:
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
        #cv2.imshow("red", cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

    def initializeK(self):
        '''
            No parameters. Returns a list of size clusterNum that represents the starting cluster centers.
            Note that it's possible to have duplicates. But this should be ok when the algorithm updates values 
        '''
        if not self.image:
            return self.data[np.random.randint(0, len(self.data), size=self.k), :]
        channels = len(self.data[0][0])
        data = np.reshape(self.data, (-1, channels))
        return data[np.random.randint(0, len(self.data), size=self.k), :]

    def assignment(self, centers):
        '''
            Returns an np array of tuples where point coordinates comprise the first element and the cluster assignment 
            is the second element for all points

            :param centers: list of cluster centers
            :type centers: np array
        '''
        if not self.image:
            return np.array([(point, np.argmin(np.array([np.sqrt(np.dot(point, mean)) for mean in centers]))) for point in self.data], dtype=object)
        channels = len(self.data[0][0])
        data = np.reshape(self.data, (-1, channels))
        return np.array([(point, np.argmin(np.array([np.sqrt(np.dot(point, mean)) for mean in centers]))) for point in data], dtype=object)        


if __name__ == '__main__':
    # applyGFilter("images/filter1_img.jpg").showcase()
    KMeansCluster("images/filter1_img.jpg", 3)

