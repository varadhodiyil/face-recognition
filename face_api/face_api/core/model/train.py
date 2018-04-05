import tensorflow as tf
from NetworkBuilder import NetworkBuilder
from DataSetGenerator import DataSetGenerator, seperateData
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import shutil
PATH =   os.path.dirname(os.path.abspath(__file__))
def diff_time(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return "%d h %d m %d s" %( t_diff.hours, t_diff.minutes, t_diff.seconds)

def clean_data(clean_images=False):
    if os.path.exists(os.path.join(PATH, "saved model v2")):
        shutil.rmtree(os.path.join(PATH, "saved model v2"))
    if os.path.exists(os.path.join(PATH, "summary_log")):
        shutil.rmtree(os.path.join(PATH, "summary_log"))
    if os.path.exists(os.path.join(PATH, "training_face_rec")) and clean_images:
        shutil.rmtree(os.path.join(PATH, "training_face_rec"))
def train():

    dg = DataSetGenerator("training_face_rec")
    MAX_LABELS  = len(dg.data_labels)
    dg.save_labels()

    with tf.name_scope("Input") as scope:
        input_img = tf.placeholder(dtype='float', shape=[None, 128, 128, 3], name="input")

    with tf.name_scope("Target") as scope:
        target_labels = tf.placeholder(dtype='float', shape=[None, MAX_LABELS], name="Targets")

    with tf.name_scope("Keep_prob_input") as scope:
        keep_prob = tf.placeholder(dtype='float',name='keep_prob')

    nb = NetworkBuilder()

    with tf.name_scope("ModelV2") as scope:
        model = input_img
        model = nb.attach_conv_layer(model, 32, summary=True)
        model = nb.attach_relu_layer(model)
        model = nb.attach_conv_layer(model, 32, summary=True)
        model = nb.attach_relu_layer(model)
        model = nb.attach_pooling_layer(model)

        model = nb.attach_conv_layer(model, 64, summary=True)
        model = nb.attach_relu_layer(model)
        model = nb.attach_conv_layer(model, 64, summary=True)
        model = nb.attach_relu_layer(model)
        model = nb.attach_pooling_layer(model)

        model = nb.attach_conv_layer(model, 128, summary=True)
        model = nb.attach_relu_layer(model)
        model = nb.attach_conv_layer(model, 128, summary=True)
        model = nb.attach_relu_layer(model)
        model = nb.attach_pooling_layer(model)

        model = nb.flatten(model)
        model = nb.attach_dense_layer(model, 200, summary=True)
        model = nb.attach_sigmoid_layer(model)
        model = nb.attach_dense_layer(model, 32, summary=True)
        model = nb.attach_sigmoid_layer(model)
        model = nb.attach_dense_layer(model, MAX_LABELS)
        prediction = nb.attach_softmax_layer(model)


    with tf.name_scope("Optimization") as scope:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=target_labels)
        cost = tf.reduce_mean(cost)
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)

    with tf.name_scope('accuracy') as scope:
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    # dg = DataSetGenerator("../data/twins")

    test_x , test_y =  dg.load_test_images(image_size=(128,128))
    epochs = 10
    batchSize = 10

    saver = tf.train.Saver()
    model_save_path="./saved model v2/"
    model_name='model'
    is_Train = True



    with tf.Session() as sess:
        summaryMerged = tf.summary.merge_all()
        
        filename = "./summary_log/run" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
        # setting global steps
        tf.global_variables_initializer().run()

        if os.path.exists(model_save_path+'checkpoint'):
            # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
        if is_Train:
            writer = tf.summary.FileWriter(filename+"_train", sess.graph)
        test_writer = tf.summary.FileWriter(filename+"_test", sess.graph)
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            
            batches = dg.get_mini_batches(batchSize,(128,128), allchannel=True)

            for imgs ,labels in batches:
                # X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.33, random_state=42)
                

                imgs=np.divide(imgs, 255)
                error, sumOut, acu, steps,_ = sess.run([cost, summaryMerged, accuracy,global_step,optimizer],
                                                feed_dict={input_img: imgs, target_labels: labels})
                writer.add_summary(sumOut, steps)
                print("epoch=", epoch,"steps=",steps, "Total Samples Trained=", steps*batchSize, "err=", error, "accuracy=", acu)
                if steps % 10 == 0:  # Record summaries and test-set accuracy
                    error, sumOut, acu, steps = sess.run([cost,summaryMerged, accuracy,global_step], feed_dict={input_img: test_x, target_labels: test_y})
                    test_writer.add_summary(sumOut, steps)
                    print("Test : error= ", error, "epoch=", epoch,"steps=",steps, "accuracy=", acu)
                if steps % 100 == 0:
                    print("Saving the model")
                    saver.save(sess, model_save_path+model_name, global_step=steps)
        saver.save(sess, model_save_path+model_name)
        sess.close()
        end_time = datetime.datetime.now()
        print(diff_time(start_time,end_time))

clean_data()
train()