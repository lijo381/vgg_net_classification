import keras
from preprocessing import preprocessing_and_labelling
from vgg_16_model import VGG16


severity_folder='/Users/leobabyjacob/Desktop/car-damage-dataset/data3a/training'
processed_data=preprocessing_and_labelling(severity_folder)
model=model = VGG16()
