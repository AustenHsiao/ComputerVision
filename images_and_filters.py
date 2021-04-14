''' 
Computer vision and deep learning. Written by Austen Hsiao
'''

'''
Part 1: Read in "filter1_img.jpg" and "filter2_img.jpg", then apply 3x3 and 5x5 Gaussian filters using 
'''

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

    def apply3Gauss(self):
        '''
            No parameters. Returns an np array of the supplied image after convolution with a 3x3 Gauss filter
        '''
        mask = np.divide(np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]), 16)
        filteredImg = []
        maskDimension = 3

        # pad 0s, width 1
        image = np.pad(self.image, 1, mode='constant', constant_values=0)

        # convolution
        filteredImg = self.convolute(mask, image)

        cv2.imshow("3x3 Gauss Filtered", filteredImg)
        cv2.imshow("originalImage", self.image)
        cv2.waitKey(0)

    def apply5Gauss(self):
        '''
            No parameters. Returns an np array of the supplied image after convolution with a 5x5 Gauss filter
        '''
        mask = np.divide(np.array([[1, 4, 7, 4, 1],
                                   [4, 16, 26, 16, 4],
                                   [7, 26, 41, 26, 7],
                                   [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]]), 273)
        filteredImg = []
        maskDimension = 5

        # pad 0s, width 2
        image = np.pad(self.image, 2, mode='constant', constant_values=0)

        # convolution
        filteredImg = self.convolute(mask, image)

        cv2.imshow("5x5 Gauss Filtered", filteredImg)
        cv2.imshow("originalImage", self.image)
        cv2.waitKey(0)

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
        return np.array([[np.dot(image[y:(y + maskDimension), x:(x + maskDimension)].flatten(), mask.flatten()) for x in range(0, xDimension - (maskDimension - 1))] for y in range(0, yDimension - (maskDimension - 1))]).astype('uint8')


if __name__ == '__main__':
    applyGFilter("images/filter2_img.jpg").apply5Gauss()
