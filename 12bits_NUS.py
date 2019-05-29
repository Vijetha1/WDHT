from keras import optimizers
import keras
from keras.layers import Dense, Dot, Dropout, Activation, Input
from keras.layers.merge import Subtract
from keras.layers.core import Lambda, Reshape, RepeatVector
from keras.models import Model
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import pdb
import sys
import utils
import numpy as np
import datetime
import json
import utils
import cv2 
import random
import h5py
from customLossFunctions import catCrossEntr, quantizationLoss, equiProbBits, dahLoss, contrastive, vectorLoss, dummy
from scipy import io as sio
from scipy.spatial.distance import cdist
import ast

from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D

lambda1 = 10.0
lambda2 = 1.0
margin =  0.2
MODEL_DIR = './../weights/weights_12bits_NUS.h5'
IMAGE_WIDTH = IMAGE_HEIGHT = 227
batch_size = 50
phase = 'Testing'
retainTop = False
nEpochs = 50
nBits = 12
nClasses = 21
totalTrainSamples = 100000
totalTestSamples = 2000
totalGallerySamples =100000

f = h5py.File('./../data/nusWide.hdf5', 'r')

trainDataImages = f['train_img']
trainDataLabels = f['train_label']
trainDataVectors = f['train_vector']


testDataImages = f['test_img']
testDataLabels = f['test_label']
testDataVectors = f['test_vector']


def test():
    galleryHashes = np.zeros((int(totalGallerySamples/batch_size)*batch_size, nBits))
    queryHashes = np.zeros((int(totalTestSamples/batch_size)*batch_size, nBits))
    galleryCls = np.zeros((int(totalGallerySamples/batch_size)*batch_size, nClasses))
    queryCls = np.zeros((int(totalTestSamples/batch_size)*batch_size, nClasses))
    DG_Gal = data_generator(totalSamples = batch_size*int(totalGallerySamples/batch_size), batch_size = batch_size, dataset='G', phase ='Test', augmentation=False, shuffle=False)
    for j in range(int(totalGallerySamples/batch_size)):
        data, lab = next(DG_Gal)
        [dummy1, h, dummy2] = multiLab.predict(data, batch_size=batch_size)
        galleryHashes[j*batch_size:(j+1)*batch_size,:] = np.asarray(h > 0.5, dtype='int32')
        galleryCls[j*batch_size:(j+1)*batch_size] = lab
        if j%batch_size == 0:
            print("Generated batch: "+str(j))

    DG_Que = data_generator(totalSamples = batch_size*int(totalTestSamples/batch_size), batch_size = batch_size, dataset = 'V', phase = 'Test', augmentation=False, shuffle=False)
    for j in range(int(totalTestSamples/batch_size)):
        data, lab = next(DG_Que)
        [dummy1, h, dummy2] = multiLab.predict(data, batch_size=batch_size)
        queryHashes[j*batch_size:(j+1)*batch_size,:] = np.asarray(h > 0.5, dtype='int32')
        queryCls[j*batch_size:(j+1)*batch_size] = lab
        if j%batch_size == 0:
            print("Generated batch: "+str(j))
    MAP = utils.getMAP(queryLabels = queryCls, databaseLabels = galleryCls, queryHashes = queryHashes, databaseHashes= galleryHashes, curType='zeroOne', typeOfData='multiLabelled')
    print("MAP for top 5000 retrieved is: "+str(MAP))

class saveWeights(Callback):
    def __init__(self):
        self.count = 0

    def on_train_begin(self, logs={}):
        test()
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass
        # multiLab.save_weights('./../pretrainedWeights/weights_12bits_NUS_epoch_'+str(self.count)+'.h5')
        # if self.count % 5 == 0 and self.count >= 0:
        #     test()
        # self.count = self.count + 1

    def on_train_end(self, logs={}):
        pass

params = {        
    "rotation_range": 6,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "shear_range": 0.2,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "fill_mode": 'reflect'
}
image_generator = ImageDataGenerator(**params)

def data_generator(totalSamples, batch_size=batch_size, dataset = 'T', phase='Train', augmentation=False, shuffle=False):
    global image_generator
    batch_count = totalSamples// batch_size  
    if dataset == 'T':
        images = trainDataImages
        vectors = trainDataVectors
        labels = trainDataLabels
    elif dataset == 'V':
        images = testDataImages
        vectors = testDataVectors
        labels = testDataLabels
    elif dataset == 'G':
        images = trainDataImages
        vectors = trainDataVectors
        labels = trainDataLabels
    while True:
        if shuffle:
            images, vectors = utils.shuffleInUnison(images, vectors)
        for i in range(batch_count):
            curBatchImages = images[i*batch_size:(i+1)*batch_size]
            curBatchVectorsTemp = vectors[i*batch_size:(i+1)*batch_size]
            curBatchLabels = labels[i*batch_size:(i+1)*batch_size]
            m_j = np.array([curBatchVectorsTemp,]*batch_size)
            m_n = np.transpose(m_j, (1, 0, 2))
            curBatchVectors = m_n - m_j
            curBatchImages = utils.cropImages(curBatchImages, cropHeight=224, cropWidth=224)
            curBatchImages = np.transpose(curBatchImages, (0, 2, 3, 1))
            # if model == 'Alexnet':
            #     curBatchImages = curBatchImages[:,::-1,:,:]
            sim = cdist(curBatchVectorsTemp,curBatchVectorsTemp ,  'cosine')
            if augmentation:
                seed = random.randint(1, 1e7)
                curBatchImages = next(image_generator.flow(curBatchImages, batch_size=batch_size, seed=seed, shuffle=False))
            if phase == 'Train':
                yield [curBatchImages, curBatchVectors], [np.zeros((batch_size, batch_size, batch_size)), np.zeros((batch_size, nBits)), sim]
            elif phase == 'Test':
                yield [curBatchImages, curBatchVectors], curBatchLabels

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
firstLayer, lastLayer = model.input, model.output

def normalize(x):
    x = K.l2_normalize(x, axis=-1)
    return x

def permuteDims(x):
    x = K.permute_dimensions(x, (1, 0, 2))
    return x

def computeDistancesforContrastive(x):
    D = K.sum(K.square(x),axis=-1)/float(nBits)
    return D

def dist(x):
    return K.sum(x, axis=-1)/float(batch_size)

vectorInput = Input(shape=(batch_size, 300))
if not retainTop:
    dense_4 = Dense(256, name='dense_4')(lastLayer)
    activ_4 = Activation('relu', name='activ_4')(dense_4)
    dense_5 = Dense(nBits, name='dense_5')(activ_4)
    output_hash = Activation('sigmoid', name='sigmoid')(dense_5)
    activ_5_rep = RepeatVector(batch_size)(output_hash)
    perm_activ_5_rep = Lambda(permuteDims)(activ_5_rep)
    activ_6_rep = RepeatVector(batch_size)(output_hash)
    mergeLayer = Subtract()([perm_activ_5_rep, activ_6_rep])
    distances = Lambda(computeDistancesforContrastive, output_shape=(batch_size,))(mergeLayer)
    dense_6 = Dense(300, name='dense_6')(activ_4)
    activ_7 = Activation('tanh', name='tanh')(dense_6)
    vectorsPred = RepeatVector(batch_size)(activ_7)
    cosine = Dot(axes=2)([vectorInput, vectorsPred])
    multiLab = Model(inputs=[firstLayer, vectorInput], outputs=[cosine, output_hash, distances])
else:
    multiLab = model

last_layer_variables = list()
multiLab_len = len(multiLab.layers)
model_len = len(model.layers)
counter = 0
for layer in multiLab.layers:
    counter = counter + 1
    if counter > model_len:
        last_layer_variables.extend(layer.weights)

# FineTuneSGD implemented using https://github.com/fchollet/keras/issues/5920

if phase == 'Testing':
    multiLab.load_weights(MODEL_DIR)
    test()
multiLab.compile(loss=[vectorLoss(l2=lambda1, m=margin), quantizationLoss(l2=lambda2, nbits=nBits), 'mean_squared_error'],
                  optimizer=optimizers.FineTuneSGD(exception_vars=last_layer_variables, lr=0.001, momentum=0.9,  multiplier=0.1))

if phase == 'Training':
    saveweights = saveWeights()
    print("Learning")
    multiLab.fit_generator(
            data_generator(totalSamples = batch_size*int(totalTrainSamples/batch_size), batch_size = batch_size, dataset = 'T'),
            steps_per_epoch=int(totalTrainSamples/batch_size),
            epochs=nEpochs,
            verbose=1,
            validation_data=data_generator(totalSamples = batch_size*int(totalTestSamples/batch_size), batch_size = batch_size, dataset='V'),
            validation_steps=int(totalTestSamples/batch_size), callbacks=[saveweights])
    multiLab.save_weights(MODEL_DIR)

