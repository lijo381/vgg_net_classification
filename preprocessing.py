import numpy as np
import os
import cv2
# severity_folder='data3a/training'
def preprocessing_and_labelling(severity_folder):
    height=width=256
    x_train = []
    y_train=[]
    label_convert=[]

    for dirs in os.listdir(severity_folder):
        if dirs.startswith('.'):
            continue
        print('*******************', dirs)
        for f in os.listdir(os.path.join(severity_folder, dirs)):
            file_p = os.path.join(severity_folder, dirs, f)
            label=file_p.split('/')[-2]
            label_final=''
            if label in label_convert:
                label_final=label_convert.index(label)
            else:
                label_convert.append(label)
                label_final = label_convert.index(label)

            img = cv2.imread(file_p)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data=cv2.resize(img,(height,width))
            img=np.rollaxis(data, 2, 0) #.reshape((data.shape[0],data.shape[1], 1))
            img_modified = img / 255.0
            # print (os.path.join(severity_folder,dirs,f))

            x_train.append(img_modified)
            y_train.append(label_final)
            # print('moderate')
    print(label_convert)
    print(y_train)
    print('Done...')
    return np.asarray(x_train),np.asarray(y_train)

