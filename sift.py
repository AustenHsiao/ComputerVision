import cv2
import numpy as np

class Sift:
    def __init__(self, file):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

        sift = cv2.SIFT_create()

        kp = sift.detect(img, None)

        #for point in kp:
        #    print(point.pt)
        #siftImg = cv2.drawKeypoints(img1, kp1, img1)
        #cv2.imshow("test", img1)
        #cv2.waitKey(0)
    
    