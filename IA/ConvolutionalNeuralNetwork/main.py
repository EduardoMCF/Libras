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

#Dividindo dados em treino e validação
images_train, images_val, labels_train, labels_val = train_test_split(images, new_labels, test_size=0.20, random_state=2)

labels_train = to_categorical(labels_train, num_classes, 'int32')
labels_val = to_categorical(labels_val, num_classes, 'int32')

train_len = len(images_train)
val_len = len(images_val)

batch_size = 32

#Definindo modelo para CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
#print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])

#Aplicando ImageDataGenerator aos dados lidos
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(images_train, labels_train, batch_size=batch_size)
val_generator = val_datagen.flow(images_val, labels_val, batch_size=batch_size)

#Treinando modelo de CNN
history = model.fit_generator(train_generator, steps_per_epoch=train_len // batch_size, epochs=50, validation_data=val_generator, validation_steps=val_len // batch_size)

#Salvando o modelo
model.save_weights(models_dir + '/model_cnn_weights_acc.h5')
model.save(models_dir + '/model_cnn_keras_acc.h5')