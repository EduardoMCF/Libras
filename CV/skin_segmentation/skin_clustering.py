import cv2
import numpy as np
from sklearn.cluster import KMeans

def skin_clustering(image):
    '''
        Receives an image in YCrCb color space as input and then apply KMeans clutering to it - with k = 2.

        Params:
            - image: An image in YCrCb color space.

        Return:
            - Returns an binary image, containing the skin segmentation.
    '''

    CbCr_pixelwise = np.array([image[x,y][1:] for x in range(image.shape[0]) for y in range(image.shape[1])])

    kmeans = KMeans(n_clusters=2).fit(CbCr_pixelwise)
    newImage = kmeans.labels_.reshape(image.shape[0],image.shape[1]).astype(np.uint8) * 255

    return newImage

