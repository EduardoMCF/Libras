import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from keras.models import load_model
from joblib import load

labels = ['1',
 '4',
 'X',
 'S',
 'B',
 'D',
 '7',
 'N',
 'V',
 'M',
 'I',
 '5',
 'T',
 'Y',
 'W',
 'C',
 'F',
 'O',
 'E',
 'A',
 '2',
 'U',
 'P',
 'G',
 'Q',
 'L',
 'R',
 '9']

def isEsc(key):
    return key & 255 == 27

def isSpace(key):
    return key & 255 == 32

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

def enhanceImage(image, structuringElement = np.ones((3,3))):
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def image_list2array_list(image_list):
  new_list = []
  for i in range(len(image_list)):
    new_image = []
    for j in range(len(image_list[i])):
      for k in range(len(image_list[i][j])):
        new_image.append(image_list[i][j][k])
    new_list.append(np.array(new_image))
  return np.array(new_list)

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

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
    newImage = kmeans.labels_.reshape(image.shape[0],image.shape[1]).astype(np.uint8)

    return newImage

def isEsc(key):
    return key & 255 == 27

def isSpace(key):
    return key & 255 == 32

def captureImagesFromWebcam(path, classifier = None):
    '''
        Initiates a video capture and loops until user end it.
        Press ESC to end the video capture.
        Press SPACE to capture an image.
        Captured images are saved within the data folder in the form:
        "capture 'counter'.png"
    '''

    cap = cv2.VideoCapture(0)
    print (cap)

    x1,y1,x2,y2 = 50, 50, 250, 250
    counter = 0
    while True:
        flag, frame = cap.read()
        x,y,_ = frame.shape
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        if not flag: break

        key = cv2.waitKey(2)
        if isEsc(key):
            print("Ending capture")
            break

        hand = cv2.resize(frame[x1+5:x2-5,y1+5:y2-5],(50,50))
        hand_YCrCb = cvt2YCrCb(hand)
        clustering = skin_clustering(hand_YCrCb)
        enhanced = enhanceImage(clustering)
        target_label = enhanced[25,25]

        if not target_label:
            enhanced = (enhanced^1)
        enhanced *= 255

        c = np.array([[[enhanced[j][i]]*3 for i in range(50)] for j in range(50)]).astype(np.uint8)

        if classifier is not None:
            if('cnn' in path):
                p = classifier.predict(c.reshape((1, 50, 50, 3)))
            elif('rf' in path):
                p = classifier.predict(c.ravel().reshape((1, -1)))
            else:
                p = classifier.predict(image_list2array_list([rgb2gray(c)]))

            label = labels[p.argmax(axis=1)[0]]
        else:
            label = 'JOOJ'

        cv2.imshow("video",frame)
        #c = cv2.integral(c)
        final_image = cv2.putText(cv2.resize(c,(300,300)), label, (150,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, 0)
        cv2.imshow('c1',final_image)

    cap.release()
    cv2.destroyAllWindows()

def test(path):

    image_RGB, image_YCrCb = load_image(path)

    binary_image = skin_clustering(image_YCrCb)

    result = enhanceImage(binary_image)

    plot_images(image_RGB,image_YCrCb,binary_image,result)

def run(path):
    model = None
    if('rf' in path):
        model = load(path)
    else:
        model = load_model(path)
    captureImagesFromWebcam(path, model)
    print("Finishing...")

run('model_rf_acc08110_no_words.h5')