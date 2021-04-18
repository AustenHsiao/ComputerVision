''' 
Computer vision and deep learning. Written by Austen Hsiao
'''
from sift import * # for part 3
from clustering import *  # for part 2
from filters import *  # for part 1
'''
class Filter(filename); a filename is supplied to the class. The filename will be for an image file
public class methods:
    showcase(savename, skip); Displays the supplied image with the various filters applied to it. Also saves all images in the current directory. No return object. savename is the filename we want to use to save. skip=whether we want to skip showing the images
    apply3Gauss(); Applies the 3x3 Gauss filter. Returns an np array
    apply5Gauss(); Applies the 5x5 Gauss filter. Returns an np array
    applyDoG(direction); A direction is supplied as a char, 'x' or 'y'. Applies the DoG filter. Returns an np array 
    applySobel(); Applies the Sobel filter using DoG in both directions. Returns an np array

class Kcluster(filename, clusternum); a filename and number of clusters is supplied to the class. The file can be data (mx2) or an image
public class methods: 
    graph(r); Runs the algorithm a total of r times, then creates a graph out of the run with the lowest sum of squares error. Prints the final SSE to console.

class Sift(); No parameters needed
public class methods:
    featureMatch(file1, file2); Creates a jpg image that draws a line between the top 10% of matched features (From file1 to file2).
'''

if __name__ == '__main__':
    #Part 1
    #Filter("images/filter1_img.jpg").showcase("filter1_img", 0)
    #Filter("images/filter2_img.jpg").showcase("filter2_img")

    #Part 2.1
    #Kcluster("data/510_cluster_dataset.txt", 2).graph(10)
    #Kcluster("data/510_cluster_dataset.txt", 3).graph(10)
    #Kcluster("data/510_cluster_dataset.txt", 4).graph(10)
    #Part 2.2
    #Kcluster("images/Kmean_img1.jpg", 5).graph(10)
    #Kcluster("images/Kmean_img1.jpg", 10).graph(2)
    Kcluster("images/Kmean_img2.jpg", 5).graph(10)
    #Kcluster("images/Kmean_img2.jpg", 10).graph(10)

    #Part 3
    #Sift().featureMatch("images/SIFT1_img.jpg", "images/SIFT2_img.jpg")
    #Sift().featureMatch("images/test1.jpg", "images/test2.jpg")

