
import os
import sys
import cv2
import numpy as np
from opencv_utils import OpenCVHelper

helper = OpenCVHelper()
def gen_train(file_in, path_out, index):

    # func_class = Func()

    """ --------------- Check the existence of input video file --------------- """
    if not os.path.isfile(file_in):
        print("Captured file not found!")
        return False

    """ ------------------- Loading and process the video file ----------------- """
    print "File %s" % file_in
    frame = cv2.imread(file_in)
    frame_ind = 0
    f_bool = True

    print("Converting " + file_in)

    f_bool = not f_bool

    # ------------ Rotate the image 90 ------------
    frame_ind += 1

    
    # cv2.imshow('image',frame)
    # cv2.waitKey(0)
    # -------------- Face detection ----------------
    img_face, _ = helper.convert_img(frame)
    # ----- Convert face image to training data ----
    if img_face is not None:
        fname_split = file_in.split('/')
        file_name = fname_split[-1].split('.')[0]
        if len(fname_split) > 1:
            user_name = file_in.split('/')[-2]
        else:
            user_name = file_name

        out_name = os.path.join(path_out, user_name, file_name + '_' + str(frame_ind) + '.bmp')
        print out_name
        helper.make_folder(out_name)
        if frame_ind % 5 ==0:
            if frame_ind % 10 == 0:
                img_face = cv2.rotate(img_face,cv2.ROTATE_90_CLOCKWISE)
            else:
                img_face = cv2.rotate(img_face,cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(out_name, img_face)

    return True

if __name__ == '__main__':

    # in_arg = ['../capture.avi',
    #           'training_face_rec']

    in_arg = ['uploads_enroll/user/video18.mp4',
              'training_face_rec']
    
    for arg_ind in range(len(sys.argv) - 1):
        in_arg[arg_ind] = sys.argv[arg_ind + 1]

    img_path = in_arg[0]
    train_path = in_arg[1]
    _,_,files = helper.get_file_list(img_path)
    for f in files:
        gen_train(f, train_path, False)

    print("Finished successfully!")
