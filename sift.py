import cv2, numpy as np
import numpy as np
import time

class Sift:
    def __init__(self):
        #img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

        self.sift = cv2.SIFT_create()
        #kp = sift.detect(img, None)

        #for point in kp:
        #    print(point.pt)
        #siftImg = cv2.drawKeypoints(img1, kp1, img1)
        #cv2.imshow("test", img1)
        #cv2.waitKey(0)
    
    def getKeyPoints(self, file):
        '''
            for the specified file, a list is returned, containing all coordinates of key points
            :param file: filename
            :type file: str
        '''
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        kp = (self.sift).detect(img, None)
        siftImg = cv2.drawKeypoints(img, kp, img)
        cv2.imshow("keypoint overlay", img)
        cv2.waitKey(0)
        return [point.pt for point in kp]

    def topTenPMatches(self, file1, file2):
        '''
            The top ten percent key point pairs are returned as a list of tuples
            :param file1: filename
            :type file1: str

            :param file2: filename
            :type file2: str
        '''
        coord1 = self.__getKeyPoints(file1)
        coord2 = self.__getKeyPoints(file2)
        #numOfMatchesToMake = len(coord1) // 10
        #counter = 0
        #pairs = []
        #start = time.time()
        #for A in coord1:
        #    if A in coord2:
        #        pairs += [[tuple(A), tuple(A)]]
        #        continue 
        #    nearIndex = np.argmin(np.array([np.linalg.norm(np.asarray(A) - np.asarray(B)) for B in coord2]))
        #    pairs += [[tuple(A), coord2[nearIndex], np.linalg.norm(np.asarray(A) - np.asarray(coord2[nearIndex]))]]
        #print(len(pairs))
        #print(len(coord1))
        #print(f'{time.time() - start} seconds')
