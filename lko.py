import numpy as np
import cv2
from filters import *


class LKO:
    def __init__(self, image1, image2):
        try:
            self.image1 = cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2GRAY)
            self.image2 = cv2.cvtColor(cv2.imread(image2), cv2.COLOR_BGR2GRAY)
        except:
            print("Error in reading image")
            exit()

        self.B = self.__calculateTemporalIntensity()
        self.xFilter = Filter(image1).applyDoG('x')
        self.yFilter = Filter(image1).applyDoG('y')

    def __calculateTemporalIntensity(self):
        '''
            Calculates the difference in intensity between 2 frames
        '''
        return np.subtract(self.image2, self.image1)

    def __calculateV(self, r, c):
        '''
        :param r: row index for the pixel
        :type r: int

        :param c: column index for the pixel
        :type c: int
        '''
        A = np.array([[self.xFilter[r-1][c-1], self.yFilter[r-1][c-1]],
                      [self.xFilter[r-1][c], self.yFilter[r-1][c]],
                      [self.xFilter[r-1][c+1], self.yFilter[r-1][c+1]],
                      [self.xFilter[r][c-1], self.yFilter[r][c-1]],
                      [self.xFilter[r][c], self.yFilter[r][c]],
                      [self.xFilter[r][c+1], self.yFilter[r][c+1]],
                      [self.xFilter[r+1][c-1], self.yFilter[r+1][c-1]],
                      [self.xFilter[r+1][c], self.yFilter[r+1][c]],
                      [self.xFilter[r+1][c+1], self.yFilter[r+1][c+1]]])

        B = np.multiply(np.array([[self.B[r-1][c-1]],
                      [self.B[r-1][c]],
                      [self.B[r-1][c+1]],
                      [self.B[r][c-1]],
                      [self.B[r][c]],
                      [self.B[r][c+1]],
                      [self.B[r+1][c-1]],
                      [self.B[r+1][c]],
                      [self.B[r+1][c+1]]]), -1)

        try:
            x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A), B))
        except:
            x = np.matmul(np.linalg.pinv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A), B))
        return (x[0][0], x[1][0])

    def __OLSimage(self):
        '''
            Calculates the OLS for all pixels in image1. Returns a list 
        '''
        col = len(self.image1[0])
        row = len(self.image1)

        return [[self.__calculateV(r, c) for c in range(1, col - 1)] for r in range(1, row - 1)]

    def __threshold(self, OF):
        '''
            Takes in a matrix of optical flow vectors; normalizes and thresholds pixels under average
            :param OF: matrix of optical flow vectors
            :type OF: np array
        '''
        OFabs = np.abs(OF)
        maximum = OFabs.max()
        normalizedOF = np.divide(OFabs, maximum)
        OFscaled = np.multiply(normalizedOF, 255)
        
        average = OFscaled.mean()
        for i in range(len(OFscaled)):
            for ii in range(len(OFscaled[0])):
                if OFscaled[i][ii] < average: 
                    OFscaled[i][ii] = 0
        return OFscaled

    def visualizeOLS(self):
        values = np.array(self.__OLSimage())
        v_x = values[:, :, 0]
        v_y = values[:, :, 1]
        n = np.sqrt(np.add(np.square(v_x), np.square(v_y)))
        v_x = self.__threshold(v_x)
        v_y = self.__threshold(v_y)
        n = self.__threshold(n)
        
        cv2.imshow('v_x', np.divide(v_x, v_x.max()/10))
        cv2.imshow('v_y', np.divide(v_y, v_y.max()/10))
        cv2.imshow('n', np.divide(n, n.max()/10))
        cv2.waitKey(0)

    
