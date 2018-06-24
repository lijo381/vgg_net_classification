from keras.utils import to_categorical

import numpy as np
from preprocessing import preprocessing_and_labelling
from vgg_16_model import VGG16,base_model
from transfer_learning import vgg_model16_pretrained
import os
nb_epoch = 1000
model_save_p='/home/ubuntu/vgg_net_classification/models_1/'
if not os.path.exists(model_save_p):
    os.mkdir(model_save_p)
# cwd=os.getcwd()
severity_folder='/home/ubuntu/car-damage-dataset/data3a/training'
severity_folder_val='/home/ubuntu/car-damage-dataset/data3a/validation'

train_x,train_y=preprocessing_and_labelling(severity_folder)
val_x,val_y=preprocessing_and_labelling(severity_folder_val)
train_y=to_categorical(train_y)
val_y=to_categorical(val_y)
# print (processed_data_val[:][0])
print(val_x.shape)
model=vgg_model16_pretrained()
# train_x,train_y=processed_data_train[:][0],processed_data_train[:][1]
# val_x,val_y=processed_data_val[:][0],processed_data_val[:][1]
# print('Train vals',train_y)
#model.compile(optimizer='adam', loss='conditional_crossentropy', metrics=['accuracy'])
# # print(train_x[0],train_y[1])
#model.fit(train_x, train_y,
#              epochs=nb_epoch, batch_size=28,verbose=1,
#              validation_data=(val_x, val_y),callbacks=callbacks_list)



from keras import callbacks

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

#early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath=model_save_p+"Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,checkpoint]

#hist = custom_vgg_model.fit(X_train, y_train, batch_size=10, epochs=1, verbose=1, validation_data=(val_data, val_label),callbacks=callbacks_list)
model.fit(train_x, train_y,
              epochs=nb_epoch, batch_size=28,verbose=1,
              validation_data=(val_x, val_y),callbacks=callbacks_list)

print('Training time: %s' % (t - time.time()))
#(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


