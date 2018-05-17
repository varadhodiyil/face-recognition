import os

import numpy as np
import tensorflow as tf
import os
import cv2
from NetworkBuilder import NetworkBuilder
from opencv_utils import OpenCVHelper

class Predictor():
    def __init__(self,dg,num_labels=3):
        self.num_labels = num_labels
        path = os.path.dirname(os.path.abspath(__file__))
        model_save_path = os.path.join(path,"saved model v2/")
        print model_save_path
        model_name='model'
        with tf.name_scope("Input") as scope:
            self.input_img = tf.placeholder(dtype='float', shape=[None, 128, 128, 3], name="input")
        self.session =  tf.Session()
            
            #     saver = tf.train.Saver()
            #     saver.restore(sess, tf.train.latest_checkpoint(model_save_path))  
                # saver = tf.train.import_meta_graph('./saved '+model_name+'/model.ckpt.meta')
                # saver = tf.train.import_meta_graph('./saved '+model_name+'/model.ckpt.meta')
                
                # saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
            # predictor =  tf.contrib.predictor.from_saved_model(model_save_path)
        if os.path.exists(model_save_path+'checkpoint'):
            print "Restoring %s " % model_name
            saver = tf.train.import_meta_graph(model_save_path+model_name+'.meta')
            saver.restore(self.session,tf.train.latest_checkpoint(model_save_path))

       
        nb = NetworkBuilder(num_labels)
        prediction = nb.get_prediction(self.input_img)


        self.y_pred_cls = tf.argmax(prediction, axis=1)
        tf.global_variables_initializer().run(session=self.session)
        label_path = os.path.join(path,"labels.pkl")
        self.labels =  dg.load_labels(path=label_path)
        self._np_labels = list()
        for i in range(len(self.labels)):
            label = np.zeros(len(self.labels),dtype=int)
            label[i] = 1
            self._np_labels.append(label)
            # print("Predicted : {}".format(labels[sample_prediction(test_img)]))
    def classify_image(self,test_im):
            h, w = test_im.shape[:2]

            sh, sw = 128 , 128
            # interpolation method
            if h > sh or w > sw:  # shrinking image
                interp = cv2.INTER_AREA
            else: # stretching image
                interp = cv2.INTER_CUBIC
            

            x = tf.placeholder(dtype="float", shape=[128, 128, 1], name='x')
            y_true = tf.placeholder(tf.float32, shape=[None, self.num_labels], name='y_true')
            # test_im = cv2.resize(test_im, (128,128), interpolation=interp)
            feed_dict_test = {
                # x: test_im.reshape(3, img_size_flat),
                # x: test_im.reshape(1, img_size_flat),
                self.input_img :test_im.reshape(-1,128,128,3),
                y_true: np.array(self._np_labels)
            }

            test_pred = self.session.run(self.y_pred_cls, feed_dict=feed_dict_test)
            return self.labels[test_pred[0]] , test_pred[0]
        
if __name__=="__main__":
    from DataSetGenerator import DataSetGenerator

    path = os.path.dirname(os.path.abspath(__file__))
    test_img = cv2.imread('3.jpeg')
    dg = DataSetGenerator(os.path.join(path,"training_face_rec"))
    MAX_LABELS  = len(dg.data_labels)
    p = Predictor(dg,num_labels=MAX_LABELS)
    helper = OpenCVHelper()
    img_face, pos_face = helper.convert_img(test_img)
    print p.classify_image(img_face)
