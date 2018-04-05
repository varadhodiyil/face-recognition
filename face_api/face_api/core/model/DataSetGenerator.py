import cv2
import numpy as np
from os.path import isfile, join
from os import listdir
from random import shuffle
from shutil import copyfile
import os
import pickle
from sklearn.model_selection import train_test_split


def seperateData(data_dir):
    for filename in listdir(data_dir):
        if isfile(join(data_dir, filename)):
            tokens = filename.split('.')
            if tokens[-1] == 'jpg':
                image_path = join(data_dir, filename)
                if not os.path.exists(join(data_dir, tokens[0])):
                    os.makedirs(join(data_dir, tokens[0]))
                copyfile(image_path, join(join(data_dir, tokens[0]), filename))
                os.remove(image_path)



class DataSetGenerator:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        if data_dir is not None:
            self.data_labels = self.get_data_labels()
            self.data_info , self.test_data = self.get_data_paths()

    def get_data_labels(self):
        data_labels = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        
        return data_labels

    def get_data_paths(self,test_size=0.10,random_state=40):
        data_paths = []
        test_paths = []
        
        for i, label in enumerate(self.data_labels):
            img_lists=[]
            path = join(self.data_dir, label)
            for filename in listdir(path):
                # tokens = filename.split('.')
                # if tokens[-1] == 'jpg':
                image_path=join(path, filename)
                img_lists.append(image_path)
            train , test = train_test_split(img_lists,test_size=test_size, random_state=random_state)
            # print train , test
            shuffle(train)
            shuffle(test)
            data_paths.append(train)
            _label = np.zeros(len(self.data_labels),dtype=int)
            _label[i] = 1 
            test_paths.append((test,_label))
        return data_paths , test_paths


    # to save the labels its optional incase you want to restore the names from the ids 
    # and you forgot the names or the order it was generated 
    def save_labels(self, path="./labels.pkl"):
        pickle.dump(self.data_labels, open(path,"wb"))

    def load_labels(self, path="./labels.pkl"):
        return pickle.load(open(path,"rb"))
         
    def get_mini_batches(self, batch_size=10, image_size=(200, 200), allchannel=True):
        images = []
        labels = []
        empty=False
        counter=0
        each_batch_size=int(batch_size/len(self.data_info))
        while True:
            for i in range(len(self.data_labels)):
                label = np.zeros(len(self.data_labels),dtype=int)
                label[i] = 1
                if len(self.data_info[i]) < counter+1:
                    empty=True
                    continue
                empty=False
                img = cv2.imread(self.data_info[i][counter])
                img = self.resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                images.append(img)

                labels.append(label)
            counter+=1

            if empty:
                break
            # if the iterator is multiple of batch size return the mini batch
            if (counter)%each_batch_size == 0:
                yield np.array(images,dtype=np.uint8), np.array(labels,dtype=np.uint8)
                del images
                del labels
                images=[]
                labels=[]

    def load_test_images(self,image_size=(200, 200),allchannel=True):
        data_x = []
        data_y = []
        for i, d in enumerate(self.test_data):
            node , label = d
            for n in node:
                img = cv2.imread(n)
                img = self.resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                
                data_x.append(img)
                data_y.append(label)
        return data_x, data_y
    def resizeAndPad(self, img, size):
        h, w = img.shape[:2]

        sh, sw = size
        # interpolation method
        if h > sh or w > sw:  # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h

        # padding
        if aspect > 1: # horizontal image
            new_shape = list(img.shape)
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img=np.zeros(new_shape, dtype=np.uint8)
            h_offset=int((w-h)/2)
            new_img[h_offset:h_offset+h, :, :] = img.copy()

        elif aspect < 1: # vertical image
            new_shape = list(img.shape)
            new_shape[0] = h
            new_shape[1] = h
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape,dtype=np.uint8)
            w_offset = int((h-w) / 2)
            new_img[:, w_offset:w_offset + w, :] = img.copy()
        else:
            new_img = img.copy()
        # scale and pad
        scaled_img = cv2.resize(new_img, size, interpolation=interp)
        return scaled_img


if __name__=="__main__":
    seperateData("./train")