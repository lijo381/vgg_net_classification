from transfer_learning import vgg_model16_pretrained
import cv2
import numpy as np
import os
from keras.models import model_from_json
import json

model_json_path = '/home/user2/Downloads/car_damage_dataset/vgg_net_classification/model.json'
weights_path = 'Best-weights-my_model-060-0.0621-0.9561.hdf5'


# json_file = open(weights_path, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
def preprocessing(img_path):
    print('Type,,,,,,,,t,ype', (img_path))
    img = None
    height = width = 256
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    # b,g,r=cv2.split(img)
    # img=cv2.merge([r,g,b])
    data = cv2.resize(img, (height, width))
    img = np.rollaxis(data, 2, 0)  # .reshape((data.shape[0],data.shape[1], 1))
    img_modified = img / 255.0
    img_reshaped = img_modified.reshape(1, 3, height, width)
    # print('image shape',img_reshaped.shape)
    return img_reshaped


model = None


def predictions(img_path):
    json_c = ''
    with open(model_json_path, 'r') as f:
        contents = f.read()
        loaded_model_json = json_c = contents
        # print(json_c)
    # print (loaded_model_json)
    img_modified = preprocessing(img_path)
    global model
    # category=img_path.split('/')[-2]
    result = ['severe', 'moderate', 'minor']
    if model == None:
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        print('model loaded...')
    res = model.predict(img_modified)[0]
    print(res)
    print('prediction>>>>>>>>>>>', result[res.argmax()])
    # if result[res.argmax()]==category:
    #   print('prediction>>>>>>>>>>>',result[res.argmax()],category)
    return result[res.argmax()]


from keras import backend as K

if __name__ == '__main__':
    # model = vgg_model16_pretrained()
    print ('Entering ..')
    print('*************************', K.image_data_format())
    folder_p = '/home/user2/Downloads/car_damage_dataset/car-damage-dataset/data3a/validation'
    correct_counter = 0
    all_files = 0
    for root, dirs, files in os.walk(folder_p):

        for f in files:
            all_files += 1
            file_p = os.path.join(root, f)
            # print(file_p)
            # processed_image=preprocessing(file_p)
            res = predictions(file_p)
            print(res)
            # if res==True:
            #     print(file_p)
            #     correct_counter+=1
            #     print('Accuracy',correct_counter/all_files)


