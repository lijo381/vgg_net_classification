from keras.models import Sequential, load_model
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
from keras import optimizers
from keras.layers import Dropout

#Get back the convolutional part of a VGG network trained on ImageNet
def vgg_model16_pretrained():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    #model_vgg16_conv.summary()

    # Create your own input format (here 3x200x200)
    input = Input(shape=(256, 256,3), name='image_input')
    #output_vgg16_conv = model_vgg16_conv(input)
    
    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)
    #dropout1 = Dropout(0.85)
    #dropout2 = Dropout(0.85)
    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc2')(x)
    #x = dropout1(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    #x = Dense(1024, activation='relu', name='fc3')(x)
    x = Dense(3, activation='softmax', name='predictions')(x)

    # Create your own model
    my_model = Model(input=input, output=x)
    for layer in my_model.layers[:-3]:
        layer.trainable = False
    my_model.summary()
    #my_model.layers[3].trainable
    my_model.compile(loss='mse',optimizer=optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),metrics=['accuracy'])
    '''top_model = Sequential()
    top_model.add(Flatten(input_shape=output_vgg16_conv.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='softmax'))

    top_model.load_weights(top_model_weights_path)  # load weights_path

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable - weights will not be updated
    for layer in model.layers[:25]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=0.000001, momentum=0.9),  # reduced learning rate by 1/10
                  metrics=['accuracy'])
    '''  
    return my_model
