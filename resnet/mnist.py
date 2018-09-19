"""
Created on Thu Aug 31 10:13:31 2018
@author: ravikalmodia
Loaded MNIST Digit Data from sklearn
Used ResNet for classification
Basic Resnet model results:
Epoch 17/50
52500/52500 [==============================] - 315s 6ms/step 
- loss: 9.7928e-04 - acc: 1.0000 - val_loss: 0.0286 - val_acc: 0.9912 
Epoch 00022: early stopping
Model fitting done. Total time: 120m 39s   
"""
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.utils import np_utils
from keras.activations import relu
from keras.layers import BatchNormalization,Conv1D,Conv2D,MaxPool2D,GlobalMaxPooling2D,Dense,Dropout,Input,Flatten,Activation
from keras.layers.convolutional import Convolution2D,ZeroPadding2D,AveragePooling2D,MaxPooling2D
from keras.callbacks import EarlyStopping,LearningRateScheduler,TensorBoard,ModelCheckpoint
from keras.optimizers import Adam,SGD
from keras.layers.merge import add,Concatenate
from keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='')
data = mnist.data
target = mnist.target
y_train = np_utils.to_categorical(target)
x_train = data / 255
x_train = x_train.reshape(70000, 28, 28,1)

#basic resnet impementation
def Resnet_model(input_tensor):
    
    def identity_block(input_tensor, kernel_size, filters):
        F1, F2, F3 = filters
        
        x = Conv2D(F1, (1, 1))(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(F2, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(F3, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = add([x, input_tensor])
        x = Activation('relu')(x)
        
        return x
    
    def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    
        F1, F2, F3 = filters
    
        x = Conv2D(F1, (1, 1), strides=strides)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        x = Conv2D(F2, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        x = Conv2D(F3, (1, 1))(x)
        x = BatchNormalization()(x)
    
        sc = Conv2D(F3, (1, 1), strides=strides)(input_tensor)
        sc = BatchNormalization()(sc)
    
        x = add([x, sc])
        x = Activation('relu')(x)
    
        return x

    net = ZeroPadding2D((3, 3))(input_tensor)
    net = Conv2D(64, (7, 7), strides=(2, 2))(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = MaxPooling2D((3, 3), strides=(2, 2))(net)

    net = conv_block(net, 3, [64, 64, 256], strides=(1, 1))
    net = identity_block(net, 3, [64, 64, 256])
    net = identity_block(net, 3, [64, 64, 256])
    
    net = AveragePooling2D((2, 2))(net)
    
    net = Flatten()(net)
    net = Dense(10, init='he_uniform',activation='softmax', name='classifier')(net)
    model = Model(input=input_tensor, output=net)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

#basic resnet50 impementation
def Resnet50_model(input_tensor):
    
    def identity_block(input_tensor, kernel_size, filters):
        F1, F2, F3 = filters
        
        x = Conv2D(F1, (1, 1))(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(F2, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(F3, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = add([x, input_tensor])
        x = Activation('relu')(x)
        
        return x
    
    def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    
        F1, F2, F3 = filters
    
        x = Conv2D(F1, (1, 1), strides=strides)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        x = Conv2D(F2, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        x = Conv2D(F3, (1, 1))(x)
        x = BatchNormalization()(x)
    
        sc = Conv2D(F3, (1, 1), strides=strides)(input_tensor)
        sc = BatchNormalization()(sc)
    
        x = add([x, sc])
        x = Activation('relu')(x)
    
        return x

    net = ZeroPadding2D((3, 3))(input_tensor)
    net = Conv2D(64, (7, 7), strides=(2, 2))(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = MaxPooling2D((3, 3), strides=(2, 2))(net)

    net = conv_block(net, 3, [64, 64, 256], strides=(1, 1))
    net = identity_block(net, 3, [64, 64, 256])
    net = identity_block(net, 3, [64, 64, 256])

    net = conv_block(net, 3, [128, 128, 512])
    net = identity_block(net, 3, [128, 128, 512])
    net = identity_block(net, 3, [128, 128, 512])
    net = identity_block(net, 3, [128, 128, 512])

    net = conv_block(net, 3, [256, 256, 1024])
    net = identity_block(net, 3, [256, 256, 1024])
    net = identity_block(net, 3, [256, 256, 1024])
    net = identity_block(net, 3, [256, 256, 1024])
    net = identity_block(net, 3, [256, 256, 1024])
    net = identity_block(net, 3, [256, 256, 1024])
    
    net = conv_block(net, 3, [512, 512, 2048])
    net = identity_block(net, 3, [512, 512, 2048])
    net = identity_block(net, 3, [512, 512, 2048])
    
    net = AveragePooling2D((2, 2))(net)
    
    net = Flatten()(net)
    net = Dense(10, init='he_uniform',activation='softmax', name='classifier')(net)
    model = Model(input=input_tensor, output=net)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

#basic resnet18 impementation
def Resnet18_model(input_tensor):

    def identity_block(input_tensor, kernel_size, filters):
        F1, F2 = filters
        
        x = Conv2D(F1, kernel_size, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(F2, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
                
        x = add([x, input_tensor])
        x = Activation('relu')(x)
        
        return x
    
    # Residual Units for increasing dimensions
    def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    
        F1, F2 = filters
    
        x = Conv2D(F1, kernel_size, strides=strides)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        x = Conv2D(F2, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        sc = Conv2D(F2, kernel_size, strides=strides)(input_tensor)
        sc = BatchNormalization()(sc)
    
        x = add([x, sc])
        x = Activation('relu')(x)
    
        return x

    net = ZeroPadding2D((3, 3))(input_tensor)
    net = Conv2D(64, (7, 7), strides=(2, 2))(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = MaxPooling2D((3, 3), strides=(2, 2))(net)

    net = conv_block(net, 3, [64, 64], strides=(1, 1))
    net = identity_block(net, 3, [64, 64])

    net = conv_block(net, 3, [128, 128])
    net = identity_block(net, 3, [128, 128])
 
    net = conv_block(net, 3, [256, 256])
    net = identity_block(net, 3, [256, 256])
    
    net = conv_block(net, 3, [512, 512])
    net = identity_block(net, 3, [512, 512])
    
    net = AveragePooling2D((2, 2))(net)
    
    net = Flatten()(net)
    net = Dense(10, init='he_uniform',activation='softmax', name='classifier')(net)
    model = Model(input=input_tensor, output=net)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def evaluate_model():
    
    def get_callbacks(patience=2):
        es = EarlyStopping('val_loss', patience=patience,verbose=1, mode="min")
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
        return [es,annealer]
    
    callbacks = get_callbacks(patience=5)
      
    X_train, X_valid, y_train, y_valid = train_test_split(x_train, target, random_state=1, test_size=0.25,stratify=target)
    
    img_input = Input(shape=(28,28,1))
    
    model = Resnet_model(img_input)
    
    start_time = time.time()
    print( 'Model fitting start ....')    
    # In a single epoch the algorithm is run with nbatches=nexamples/batchsize
    model.fit(X_train, y_train,
              batch_size=256,
              nb_epoch=32,
              verbose=2,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks
              )
    
    m, s = divmod( time.time() - start_time, 60 )
    print( 'Model fitting done. Total time: {}m {}s'.format(int(m), int(s)) )
    
    #Loss curve uncomment to plot graph
#    plt.figure(figsize=[8,6])
#    plt.plot(model.history['loss'],'r',linewidth=3.0)
#    plt.plot(model.history['val_loss'],'b',linewidth=3.0)
#    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
#    plt.xlabel('Epochs ',fontsize=16)
#    plt.ylabel('Loss',fontsize=16)
#    plt.title('Loss Curves',fontsize=16)
#    plt.savefig('Loss_curve.png') 
    
    # Accuracy Curves
#    plt.figure(figsize=[8,6])
#    plt.plot(model.history['acc'],'r',linewidth=3.0)
#    plt.plot(model.history['val_acc'],'b',linewidth=3.0)
#    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
#    plt.xlabel('Epochs ',fontsize=16)
#    plt.ylabel('Accuracy',fontsize=16)
#    plt.title('Accuracy Curves',fontsize=16)
#    plt.savefig('Accuracy_curve.png') 
    

evaluate_model()

