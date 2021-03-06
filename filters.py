import cv2, numpy as np, random



class Filter:
    def __init__(self, file):
        try:
            self.image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        except:
            print("Error reading file.")
            exit()
        self.x = len(self.image[0])
        self.y = len(self.image)

    def showcase(self, savename, skip=1):
        '''
            Displays image results for all filters
            :param savename: name to save the files as
            :type savename: str

            :param skip: do we want to skip showing the images? 1=yes, 0=no. Deafult yes
            :type skip: int
        '''
        filtered1 = np.multiply(self.apply3Gauss(), 255)
        filtered2 = np.multiply(self.apply5Gauss(), 255)
        filtered3 = np.multiply(self.applyDoG('x'), 255)
        filtered4 = np.multiply(self.applyDoG('y'), 255)
        filtered5 = np.multiply(self.applySobel(), 255)

        if not cv2.imwrite(f"Gauss_3_{savename}.jpg", filtered1) or not cv2.imwrite(f"Gauss_5_{savename}.jpg", filtered2)\
            or not cv2.imwrite(f"DoG_x_{savename}.jpg", filtered3) or not cv2.imwrite(f"DoG_y_{savename}.jpg", filtered4)\
                or not cv2.imwrite(f"Sobel_{savename}.jpg", filtered5):
                print("Something went wrong when trying to save file...")

        if not skip:
            filtered1 = self.apply3Gauss()
            filtered2 = self.apply5Gauss()
            filtered3 = self.applyDoG('x')
            filtered4 = self.applyDoG('y')
            filtered5 = self.applySobel()

            cv2.imshow("originalImage", self.image)
            cv2.imshow("3x3 Gauss Filtered", filtered1)
            cv2.imshow("5x5 Gauss Filtered", filtered2)
            cv2.imshow(f"Derivative of Gauss (x) Filtered", filtered3)
            cv2.imshow(f"Derivative of Gauss (y) Filtered", filtered4)
            cv2.imshow(f"Sobel Filtered", filtered5)
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
        return self.__convolute(mask, image)

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
        return self.__convolute(mask, image)

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
        return self.__convolute(mask, image)

    def applySobel(self):
        '''
            No parameters. Returns an np array of the supplied image after convolution with the derivative of Gauss
            filters to produce the Sobel filtered result
        '''
        return np.sqrt(np.add(np.square(self.applyDoG('x')), np.square(self.applyDoG('y'))))

    def __convolute(self, mask, image):
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
