''' 
Computer vision and deep learning. Written by Austen Hsiao
'''
from filters import *  # for part 1
'''
class Filter(filename); a filename is supplied to the class. The filename will be for an image file
public class methods:
    showcase(); Displays the supplied image with the various filters applied to it. No return object
    apply3Gauss(); Applies the 3x3 Gauss filter. Returns an np array
    apply5Gauss(); Applies the 5x5 Gauss filter. Returns an np array
    applyDoG(direction); A direction is supplied as a char, 'x' or 'y'. Applies the DoG filter. Returns an np array 
    applySobel(); Applies the Sobel filter using DoG in both directions. Returns an np array
'''
from clustering import *  # for part 2
'''
class Kcluster(filename, clusternum); a filename and number of clusters is supplied to the class. The file can be data (mx2) or an image
public class methods: 
    graph(r); Runs the algorithm a total of r times, then creates a graph out of the run with the lowest sum of squares error
'''

if __name__ == '__main__':
    # applyGFilter("images/filter1_img.jpg").showcase()
    #Kcluster("data/510_cluster_dataset.txt", 8).graph(10)
    Kcluster("images/small.jpg", 5).graph(1)