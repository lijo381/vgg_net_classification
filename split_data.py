import os
import shutil





class split_test_train_val():

    def __init__(self,source_dataset,train_val_split_ratio=0.3,val_test_split_ratio=0.3):
        self.dataset=source_dataset
        self.currentworkingdir=os.getcwd()
        self.dest_dataset = os.path.join(self.currentworkingdir, '..', 'target_dataset_folder')
        self.train_dataset = os.path.join(self.dest_dataset, 'train/')
        self.test_dataset = os.path.join(self.dest_dataset, 'test/')
        self.val_dataset = os.path.join(self.dest_dataset, 'validation/')
        self.train_val_split = train_val_split_ratio
        self.val_test_split=val_test_split_ratio



    def make_dirs_first_time(self):
        if not os.path.exists(self.dest_dataset):
            os.mkdir(self.dest_dataset)
            os.mkdir(self.train_dataset)
            os.mkdir(self.test_dataset)
            os.mkdir(self.val_dataset)



    def create_dirs(self,paths):

        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)



    def replicate_folders(self):
        self.make_dirs_first_time()

        for root,dir,files in os.walk(self.dataset):
            val_list = test_list = []
            print (root,len(files))
            train_path=root.replace(self.dataset,self.train_dataset)
            val_path=root.replace(self.dataset,self.val_dataset)
            test_path=root.replace(self.dataset,self.test_dataset)
            train_data_count,val_data_count = int(len(files)*(1-self.train_val_split)),int(len(files)*(self.train_val_split))
            val_final,test_final=val_data_count*(1-self.val_test_split),val_data_count*(self.val_test_split)
            val_final=int(val_final)
            test_final=int(test_final)
            # print ('Length',train_data_count,val_final,test_final)
            all_folders=[train_path,val_path,test_path]
            self.create_dirs(all_folders)
            counter_train=counter_val=counter_test=0

            for file in files:

                file_path=os.path.join(root,file)
                if len(val_list)<=val_data_count:
                    if  len(test_list)<=test_final:
                        shutil.copy(file_path, os.path.join(test_path, file))
                        test_list.append(file_path)
                        counter_test += 1
                    else:

                        shutil.copy(file_path,os.path.join(val_path,file))
                        val_list.append(file_path)
                        counter_val += 1
                else:
                    shutil.copy(file_path, os.path.join(train_path, file))
                    counter_train+=1
            print('------------------>',counter_train, counter_val, counter_test)







if __name__=='__main__':
    source_dataset = '/home/user2/car_orientation_detetction/ImageRecognitionDataSet/'
    obj1=split_test_train_val(source_dataset).replicate_folders()






