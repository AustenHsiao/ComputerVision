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
    
    def __getDesc(self, file):
        '''
            for the specified file, the keypoint descriptors are returned. Saves an image
            containing the overlayed keypoints
            :param file: filename
            :type file: str
        '''
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        kp, desc = (self.sift).detectAndCompute(img, None)
        siftImg = cv2.drawKeypoints(img, kp, img)       
        filename = f"kp_{file.split('/')[1]}"
        if not cv2.imwrite(filename, img):
            print(f"Could not save overlayed keypoints. Filename: {filename} cannot be used")
        return (kp, desc)

    def featureMatch(self, file1, file2):
        '''
            The top ten percent key point pairs are matched and visualized. Saves an image
            containing lines between the two images corresponding to matched features.
            :param file1: filename
            :type file1: str

            :param file2: filename
            :type file2: str
        '''
        img1 = cv2.cvtColor(cv2.imread(file1), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(file2), cv2.COLOR_BGR2RGB)

        r1 = self.__getDesc(file1)
        r2 = self.__getDesc(file2)
        kp1 = r1[0] 
        kp2 = r2[0]
        d1 = r1[1]
        d2 = r2[1]

        topTenPercent = len(d1) // 10 

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(d1,d2)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:topTenPercent], None, flags=2)
        if not cv2.imwrite("Matches.jpg",img3):
            print("Could not save matches for some unknown reason")