
import os
import sys
import cv2
import numpy as np
from opencv_utils import OpenCVHelper
func_class = OpenCVHelper()


def gen_train(file_in, path_out, rot_enable=False):


    """ --------------- Check the existence of input video file --------------- """
    if not os.path.isfile(file_in):
        print("Captured file not found!")
        return False

    """ ------------------- Loading and process the video file ----------------- """
    cap = cv2.VideoCapture(file_in)
    frame_ind = 0
    f_bool = True

    print("Converting " + file_in)

    while True:
        ret, frame = cap.read()
        print("   Converting frame" + str(frame_ind))
        if not ret:
            break

        f_bool = not f_bool

        if f_bool:
            # ------------ Rotate the image 90 ------------
            frame_ind += 1

            if rot_enable:
                frame = np.rot90(frame)
            # cv2.imshow('image',frame)
            # cv2.waitKey(0)
            # -------------- Face detection ----------------
            img_face, _ = func_class.convert_img(frame)
            # ----- Convert face image to training data ----
            if img_face is not None:
                fname_split = file_in.split('/')
                file_name = fname_split[-1].split('.')[0]
                if len(fname_split) > 1:
                    user_name = file_in.split('/')[-2]
                else:
                    user_name = file_name
                
                out_name = os.path.join(path_out, user_name, file_name + '_' + str(frame_ind) + '.bmp')
                func_class.make_folder(out_name)
                if frame_ind % 3 ==0:
                    if frame_ind % 6 == 0:
                        img_face = cv2.rotate(img_face,cv2.ROTATE_90_CLOCKWISE)
                    else:
                        img_face = cv2.rotate(img_face,cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(out_name, img_face)

    return True

if __name__ == '__main__':
    BASE_DIR = os.path.dirname((os.path.abspath(__file__)))

    # in_arg = ['../capture.avi',
    #           'training_face_rec']
    # MEDIA_ROOT = os.path.join(BASE_DIR, 'face_api','core','model','media')
    BASE_DIR = os.path.join(BASE_DIR,"media")
    
    _, _ ,files = func_class.get_file_list(BASE_DIR)
    for f in files:
        gen_train(f,"training_face_rec")
    # in_arg = ['uploads_enroll/user/video18.mp4',
    #           'training_face_rec']

    # for arg_ind in range(len(sys.argv) - 1):
    #     in_arg[arg_ind] = sys.argv[arg_ind + 1]

    # avi_file = in_arg[0]
    # train_path = in_arg[1]

    # gen_train(avi_file, train_path, False)

    # print("Finished successfully!")
