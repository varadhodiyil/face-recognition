import tensorflow as tf
import os
import cv2
import numpy as np
from NetworkBuilder import NetworkBuilder
from DataSetGenerator import DataSetGenerator
# saver = tf.train.Saver()
model_save_path="saved model v2/"
model_name='model'
with tf.name_scope("Input") as scope:
    input_img = tf.placeholder(dtype='float', shape=[None, 128, 128, 1], name="input")
with tf.Session() as sess:
    # if os.path.exists(model_save_path+'checkpoint'):
    #     saver = tf.train.Saver()
    #     saver.restore(sess, tf.train.latest_checkpoint(model_save_path))  
        # saver = tf.train.import_meta_graph('./saved '+model_name+'/model.ckpt.meta')
        # saver = tf.train.import_meta_graph('./saved '+model_name+'/model.ckpt.meta')
         
        # saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
    # predictor =  tf.contrib.predictor.from_saved_model(model_save_path)
    saver = tf.train.import_meta_graph(model_save_path+model_name+'.meta')
    saver.restore(sess,tf.train.latest_checkpoint(model_save_path))

# import tensorflow as tf
# import os
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# # Create some variables.
# # v1 = tf.Variable([11.0, 16.3], name="Activation")

# model_save_path="./saved model v2/"
# model_name='model'
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
# if os.path.exists(model_save_path+'checkpoint'):
#     # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
#     saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# # Restore variables from disk.
# # ckpt_path = model_save_path+model_name
# # saver.restore(sess, ckpt_path + '-'+ str(1))
# print("Model restored.")

# # print sess.run(v1)
# # print sess.run(v2)

    nb = NetworkBuilder()
    prediction = nb.get_prediction(input_img)


    y_pred_cls = tf.argmax(prediction, axis=1)
    tf.global_variables_initializer().run()
    
    def sample_prediction(test_im,_labels_):
            h, w = test_im.shape[:2]

            sh, sw = 128 , 128
            # interpolation method
            if h > sh or w > sw:  # shrinking image
                interp = cv2.INTER_AREA
            else: # stretching image
                interp = cv2.INTER_CUBIC
            num_channels = 1

            # image dimensions (only squares for now)
            img_size = 128

            # Size of image when flattened to a single dimension
            img_size_flat = img_size * img_size * num_channels

            x = tf.placeholder(dtype="float", shape=[None, 128, 128, 1], name='x')
            y_true = tf.placeholder(tf.float32, shape=[None, 3], name='y_true')
            test_im = cv2.resize(test_im, (128,128), interpolation=interp)
            feed_dict_test = {
                # x: test_im.reshape(3, img_size_flat),
                # x: test_im.reshape(1, img_size_flat),
                input_img :test_im.reshape([-1, 128,128, 1]),
                y_true: np.array(_labels_)
            }

            test_pred = sess.run(y_pred_cls, feed_dict=feed_dict_test)
            return test_pred[0]
    test_img = cv2.imread('temp.jpg')
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    dg = DataSetGenerator("train_faces")
    labels =  dg.load_labels()
    _np_labels = list()
    for i in range(len(labels)):
        label = np.zeros(len(labels),dtype=int)
        label[i] = 1
        _np_labels.append(label)
    print("Predicted : {}".format(labels[sample_prediction(test_img,_np_labels)]))
