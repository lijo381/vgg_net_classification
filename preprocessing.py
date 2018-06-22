import os
import cv2
import numpy as np
# severity_folder='data3a/training'
def preprocessing_and_labelling(severity_folder):
    height=width=224
    x_train = []
    y_train=[]
    for dirs in os.listdir(severity_folder):

        print('*******************', dirs)
        for f in os.listdir(os.path.join(severity_folder, dirs)):
            file_p = os.path.join(severity_folder, dirs, f)
            img = cv2.imread(file_p)
            img=cv2.resize(img,(height,width))
            img_modified = img / 255.0
            # print (os.path.join(severity_folder,dirs,f))
            if 'moderate' in dirs:
                x_train.append(img_modified)
                y_train.append(np.asarray([0,0,1]))
                # print('moderate')
            elif 'minor' in dirs:
                # print('minor')
                x_train.append(img_modified)
                y_train.append( np.asarray([0,1,0]))
            elif 'severe' in dirs:
                # print('severe')
                x_train.append(img_modified)
                y_train.append(np.asarray([1,0,0]))

    print('Done...')
    return np.asarray(x_train),np.asarray(y_train)