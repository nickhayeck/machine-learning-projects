# Machine Learning Homework 4 - Image Classification

__author__ = 'ngh6bax'
GRADINGMODE = True


# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys
import pandas as pd
import tensorflow as tf
from time import gmtime, strftime


# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

### HELPER FUNCTIONS ###
def get_data(datafile):
  dataframe = pd.read_csv(datafile)
  dataframe = shuffle(dataframe)
  data = list(dataframe.values)
  labels, images = [], []
  for line in data:
    labels.append(line[0])
    images.append(line[1:])
  labels = np.array(labels)
  images = np.array(images).astype('float32')
  images /= 255
  return images, keras.utils.to_categorical(labels, num_classes=10)

def visualize_weights(trained_model, num_to_display=2, save=True, hot=True):
  layer1 = trained_model.layers[0]
  weights = layer1.get_weights()[0]

  # Feel free to change the color scheme
  colors = 'hot' if hot else 'binary'

  for i in range(num_to_display):
    wi = weights[:,i].reshape(28, 28)
    plt.imshow(wi, cmap=colors, interpolation='nearest')
    plt.show()

def output_predictions(predictions):
  with open('predictions.txt', 'w+') as f:
    for pred in predictions:
      f.write(str(np.argmax(pred)) + '\n')

def plot_history(history):
  train_loss_history = history.history['loss']
  val_loss_history = history.history['val_loss']

  train_acc_history = history.history['accuracy']
  val_acc_history = history.history['val_accuracy']

  # plot
  plt.plot(train_loss_history)
  plt.plot(val_loss_history)
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  plt.plot(train_acc_history)
  plt.plot(val_acc_history)
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

### MULTILAYER PERCEPTRON ###
def mlp(x_train, y_train, args=None):
    # Define model architecture
    model = keras.Sequential([
                                keras.layers.Input(shape=(784)),
                                keras.layers.Dense(128),
                                keras.layers.Dense(128),
                                keras.layers.Dense(10),
                                keras.layers.Activation(keras.activations.softmax)
                                ])

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=args['LR'])

    # Compilation
    model.compile(optimizer=opt,
              	loss=keras.losses.CategoricalCrossentropy(),
              	metrics=['accuracy'])

    #Training
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=args['epochs'],
                        validation_split=0.1)
    return model, history

### CONVOLUTIONAL NEURAL NETWORK ###
def cnn(x_train, y_train, args=None):
    x_train = x_train.reshape(60000,28,28,1)
    #structure
    model=keras.Sequential([
                            keras.layers.Conv2D(64,(3,3), padding='same', input_shape=(28,28,1)),
                            keras.layers.Activation(keras.activations.relu),
                            keras.layers.MaxPooling2D((2,2)),
                            keras.layers.Flatten(),
                            keras.layers.Dense(128),
                            keras.layers.Activation(keras.activations.relu),
                            keras.layers.Dense(10),
                            keras.layers.Activation(keras.activations.softmax)
    ])

    #optimizer
    opt = keras.optimizers.Adam(learning_rate=args['LR'])

    #compile
    model.compile(optimizer=opt,
              	loss=keras.losses.CategoricalCrossentropy(),
              	metrics=['accuracy'])

    #training
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=args['epochs'])
    return model, history

### LOGISTIC REGRESSION ###
def logreg(x_train, y_train, args=None):
    # Define model architecture
    model = keras.Sequential([
                                keras.layers.Input(shape=(args['comp'])),
                                keras.layers.Dense(10),
                                keras.layers.Activation(keras.activations.softmax)
                                ])

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=args['LR'])

    # Compilation
    model.compile(optimizer=opt,
              	loss='categorical_crossentropy',
              	metrics=['accuracy'])

    #Training
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=args['epochs'],
                        validation_split=0.1)
    return model, history

### PRINCIPAL COMPONENT ANALYSIS ###
def pca(x_train, components=150):
  scaler = StandardScaler()
  scaler.fit(x_train)
  x_sc_train = scaler.transform(x_train)
  pca = PCA(n_components=components)
  pca.fit(x_sc_train)
  x_train_pca=pca.fit_transform(x_train)
  return x_train_pca

def build_model(hyp_params, type='mlp'):
    x_train, y_train = get_data(hyp_params['train_fp'])

    if type == 'mlp':
        model, history = mlp(x_train, y_train, args=hyp_params)

    elif type == 'cnn':
        model, history = cnn(x_train, y_train, args=hyp_params)
    
    elif type == 'log':
        model, history = logreg(x_train, y_train, args=hyp_params)
    elif type == 'logpca':
        x_train = pca(x_train,components=hyp_params['comp'])
        model, history = logreg(x_train, y_train, args=hyp_params)
    
    if not GRADINGMODE:
      print(model.summary())
      plot_history(history)
      visualize_weights(model)
    
    return model, history



if __name__ == '__main__':
    
    if GRADINGMODE:
      if (len(sys.argv) != 3):
        print("Usage:\n\tpython3 fashion.py train_file test_file")
        exit()
      train_file, test_file = sys.argv[1], sys.argv[2]
    else:
      train_file, test_file = r'C:\Users\HayeckFamily\Documents\dont touch\recovered files\SMH\Documents\Nick\skule\Y2T1\CS4774\hw3\fashion_train.csv', ''

    #Define hyperparameters
    hyp_params = {'LR' : 0.001,
                  'epochs' : 25,
                  'train_fp' : train_file,
                  'test_fp' : test_file,
                  'comp' : 350
                }
    model, history = build_model(hyp_params, type='cnn')
    
    if GRADINGMODE:
      x_test,_= get_data(test_file)
      x_test = x_test.reshape(-1,28,28,1)
      predictions = model.predict(x_test)
      output_predictions(predictions)
