import matplotlib.pyplot as plt
import os
import random
import gc
from glob import glob
from keras.preprocessing.image import load_img
from itertools import chain
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical
import numpy as np

data_dir = '../Data/'
models_dir = '../Models/'

def load_dataset(path : str = '', labels : [str] = ['A','B','C','D','E','F','L']) -> ([[int]],[str]):
  path = os.path.join(path,"Folds_Dataset_Final/*/")
  paths = {label:glob(f"{path}{label}/c*.PNG") for label in labels}
  data = {label:list(map(lambda path: load_img(path), pathList)) for label, pathList in paths.items()}

  x = np.array(list(chain(*[list(map(lambda x: np.array(x),value)) for value in data.values()])))
  y = np.array(list(chain(*[[k]*len(v) for k,v in data.items()])))

  return x,y,paths

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

#Lendo dados do dataset
images, labels, _ = load_dataset(data_dir)

num_classes = len(set(labels))
print('number of classes: %d'%num_classes)

#Trocando os labels da lista por inteiros
labels_set = list(set(labels))
labels_map_num_to_cat = {}
labels_map_cat_to_num = {}

for i in range(num_classes):
  labels_map_num_to_cat[i] = labels_set[i]
  labels_map_cat_to_num[labels_set[i]] = i

new_labels = []
for label in labels:
  new_labels.append(labels_map_cat_to_num[label])

#Dividindo dados em treino, validação e teste
images_train, images_test, labels_train, labels_test = train_test_split(images, new_labels, test_size=0.20, random_state=7, shuffle=True)
images_train, images_val, labels_train, labels_val = train_test_split(images_train, labels_train, test_size=0.20, random_state=7, shuffle=True)

labels_train = to_categorical(labels_train, num_classes, 'int32')
labels_val = to_categorical(labels_val, num_classes, 'int32')
labels_test = to_categorical(labels_test, num_classes, 'int32')
labels = to_categorical(new_labels, num_classes, 'int32')

test_len = len(images_test)
train_len = len(images_train)
val_len = len(images_val)

batch_size = 32

#Convertendo imagens em arrays unidimensionais
images_gray = list(map(rgb2gray, images))
images_train_gray = list(map(rgb2gray, images_train))
images_val_gray = list(map(rgb2gray, images_val))
images_test_gray = list(map(rgb2gray, images_test))

images_gray = image_list2array_list(images_gray)
images_train_gray = image_list2array_list(images_train_gray)
images_val_gray = image_list2array_list(images_val_gray)
images_test_gray = image_list2array_list(images_test_gray)

#Definindo modelo
model_mlnn = models.Sequential()
model_mlnn.add(layers.Dense(units=50, input_dim=2500, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu'))
model_mlnn.add(layers.Dense(units=50, input_dim=50, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh'))
model_mlnn.add(layers.Dense(units=7, input_dim=50, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax'))

sgd_optimizer = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9)

model_mlnn.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['acc'])
#print(model_mlnn.summary())


history = model_mlnn.fit(images_train_gray, labels_train, batch_size=batch_size, epochs=100, validation_data=(images_val_gray, labels_val))

model_mlnn.save_weights(models_dir + '/model_mlnn_weights_acc09870.h5')
model_mlnn.save(models_dir + '/model_mlnn_keras_acc09870.h5')