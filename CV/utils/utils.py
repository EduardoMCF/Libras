import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..skin_segmentation.skin_clustering import skin_clustering

import re
from glob import glob

def isEsc(key):
    return key & 255 == 27

def isSpace(key):
    return key & 255 == 32

def captureImagesFromWebcam():
    '''
        Initiates a video capture and loops until user end it.

        Press ESC to end the video capture.
        Press SPACE to capture an image.

        Captured images are saved within the data folder in the form:
        "capture 'counter'.png"
    '''

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    files = glob("./CV/data/capture [0-9].png")
    numbers = [int(re.search("[0-9]{1,}",file).group()) for file in files]
    counter = max(numbers)+1 if numbers else 0

    while True:
        flag, frame = cap.read()

        if not flag: break

        cv2.imshow("video",frame)

        key = cv2.waitKey(2)
        if isEsc(key):
            print("Ending capture")
            break
        elif isSpace(key):
            img_Name = f"capture {counter}.png"
            cv2.imwrite(f"./CV/data/{img_Name}", frame)
            counter += 1
            print("Sucessfully captured image")

    cap.release()
    cv2.destroyAllWindows()

def cvt2YCrCb(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)

def cvt2RGB(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def load_image(path):
    image = cv2.imread(path)
    return cvt2RGB(image), cvt2YCrCb(image)

def plot_images(*args):
    fig, ax = plt.subplots(len(args),1, figsize=(30,20))
    for i, image in enumerate(args):
        ax[i].imshow(image, interpolation="bilinear")
        ax[i].axis("off")
    plt.show()

def enhanceImage(image, structuringElement = np.ones((5,5))):
    ''' Enhance a binary image by apllying a closing, then a opening and finally a gaussian blur.
        Params:
            - image : The input binary image.
            - structuringElement : The structuring element used in morphology.

        Obs: Structuring element must have an odd size.
    '''

    assert structuringElement.shape[0] & 1 and structuringElement.shape[1] & 1

    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, structuringElement)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, structuringElement)

    blur = cv2.GaussianBlur(opening,structuringElement.shape,100)

    return blur

def test(path):

    image_RGB, image_YCrCb = load_image(path)

    binary_image = skin_clustering(image_YCrCb)

    result = enhanceImage(binary_image)

    plot_images(image_RGB,image_YCrCb,binary_image,result)



