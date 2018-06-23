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
    input = Input(shape=(3, 224, 224), name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)
    #dropout1 = Dropout(0.85)
    #dropout2 = Dropout(0.85)
    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(512, activation='relu', name='fc1')(x)
    #x = dropout1(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    #x = Dense(1024, activation='relu', name='fc3')(x)
    x = Dense(3, activation='softmax', name='predictions')(x)

    # Create your own model
    my_model = Model(input=input, output=x)
    for layer in my_model.layers[:-1]:
        layer.trainable = False
    my_model.summary()
    my_model.layers[3].trainable
    my_model.compile(loss='mse',optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),metrics=['accuracy'])
    return my_model
