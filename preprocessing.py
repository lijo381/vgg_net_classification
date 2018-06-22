import os
import cv2
# severity_folder='data3a/training'
def preprocessing_and_labelling(severity_folder):
    height=width=224
    data_list = []
    for dirs in os.listdir(severity_folder):

        print('*******************', dirs)
        for f in os.listdir(os.path.join(severity_folder, dirs)):
            file_p = os.path.join(severity_folder, dirs, f)
            img = cv2.imread(file_p)
            img=cv2.resize(img,(height,width))
            img_modified = img / 255.0
            # print (os.path.join(severity_folder,dirs,f))
            if 'moderate' in dirs:
                data_list.append([img_modified, 'moderate'])
                # print('moderate')
            elif 'minor' in dirs:
                # print('minor')
                data_list.append([img_modified, 'minor'])
            elif 'severe' in dirs:
                # print('severe')
                data_list.append([img_modified, 'severe'])

    print('Done...')
    return data_list