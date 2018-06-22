import numpy as np
from preprocessing import preprocessing_and_labelling
from vgg_16_model import VGG16

nb_epoch = 1

severity_folder='/home/user2/Downloads/car_damage_dataset/car-damage-dataset/data3a/training'
severity_folder_val='/home/user2/Downloads/car_damage_dataset/car-damage-dataset/data3a/validation'

train_x,train_y=preprocessing_and_labelling(severity_folder)
val_x,val_y=preprocessing_and_labelling(severity_folder_val)
# print (processed_data_val[:][0])

model=model = VGG16()
# train_x,train_y=processed_data_train[:][0],processed_data_train[:][1]
# val_x,val_y=processed_data_val[:][0],processed_data_val[:][1]
# print('Train vals',train_y)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # print(train_x[0],train_y[1])
model.fit(train_x, train_y,
              epochs=nb_epoch, batch_size=2,verbose=1,
              validation_data=(val_x, val_y))