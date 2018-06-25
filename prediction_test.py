
from transfer_learning import vgg_model16_pretrained
import cv2
import numpy as np
import os


weights_path='/home/ubuntu/vgg_net_classification/models_1/Best-weights-my_model-019-0.1177-0.8080.hdf5'
def preprocessing(img_path):
    height = width = 256
    img=cv2.imread(img_path)
    b,g,r=cv2.split(img)
    img=cv2.merge([r,g,b])
    data = cv2.resize(img, (height, width))
    img = data#np.rollaxis(data, 2, 0)  # .reshape((data.shape[0],data.shape[1], 1))
    img_modified = img/ 255.0
    img_reshaped=img_modified.reshape(1,height,width,3)
    #print('image shape',img_reshaped.shape)
    return img_reshaped
model=None
def predictions(img_path,img_modified):
    global model
    category=img_path.split('/')[-2]
    result=['03-severe', '02-moderate', '01-minor']
    if model==None:
        model=vgg_model16_pretrained()
        model.load_weights(weights_path)
        print('model loaded...')
    res=model.predict(img_modified)[0]
    #print(res)
    if result[res.argmax()]==category:
      print('prediction>>>>>>>>>>>',result[res.argmax()],category)
    return result[res.argmax()]==category



if __name__=='__main__':
    # model = vgg_model16_pretrained()
    folder_p='/home/ubuntu/car-damage-dataset/data3a/validation'
    correct_counter=0
    all_files=0
    for root,dirs,files in os.walk(folder_p):
        
        for f in files:
            all_files+=1
            file_p=os.path.join(root,f)
            #print(file_p)
            processed_image=preprocessing(file_p)
            res=predictions(file_p,processed_image)
            if res==True:
                print(file_p)
                correct_counter+=1
                print('Accuracy',correct_counter/all_files)


