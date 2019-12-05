import os
from glob import glob
from keras.preprocessing.image import load_img
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_dataset(path : str = '', labels : [str] = ['A','B','C','D','E','F','L']) -> ([[int]],[str]):
  '''Loads dataset, returning a tuple with data, label and data paths'''
  
  path = os.path.join(path,"Folds_Dataset_Final/*/")
  paths = {label:glob(f"{path}{label}/c*.PNG") for label in labels}
  data = {label:list(map(lambda path: load_img(path, color_mode='grayscale'), pathList)) for label, pathList in paths.items()}

  x = np.array(list(chain(*[list(map(lambda x: np.array(x),value)) for value in data.values()])))
  y = np.array(list(chain(*[[k]*len(v) for k,v in data.items()])))

  return x,y,paths

  # load dataset and check if it worked
path = '../Data/'
x, y, _ = load_dataset(path)
display(x.shape)
plt.imshow(X[80])