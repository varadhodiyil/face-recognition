import cv2
import dlib
import os

class OpenCVHelper(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        path = os.path.dirname(os.path.abspath(__file__))
        self.predictor = dlib.shape_predictor(
            # 'models/shape_predictor_68_face_landmarks.dat'
            os.path.join(path,'models/dlib_model.dat'))

    def get_landmarks(self, im):
        rect = self.detector(im, 0)

        if len(rect) != 1:  # raise TooManyFaces or NoFaces
            return

        points = []
        predict_ret = self.predictor(im, rect[0]).parts()
        for p in predict_ret:
            points.append((p.x, p.y))

        return points

    def get_rect(self, p_landmarks):
        p_x = []
        p_y = []
        for point in p_landmarks:
            p_x.append(point[0])
            p_y.append(point[1])

        return min(p_x), max(p_x), min(p_y), max(p_y)

    def get_face_img(self, im):
        landmarks = self.get_landmarks(im)

        if landmarks is None:
            return None, None

        min_x, max_x, min_y, max_y = self.get_rect(landmarks)
        len_x = max_x - min_x
        len_y = max_y - min_y
        _, width = im.shape[:2]
        im_face = im[max(min_y - len_y / 3, 0):max_y,
                     max(min_x - len_x / 8, 0):min(max_x + len_x / 8, width)]

        return im_face, [min_x, max_x, min_y, max_y]

    def convert_img(self, img_in):
        img_face, pos_face = self.get_face_img(img_in)
        if img_face is None:
            return None, None
        else:
            img_resize = cv2.resize(
                img_face, (128, 128), interpolation=cv2.INTER_AREA)
            return img_resize, pos_face

    def convert_to_greyscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def make_folder(self, file_path):

        path_out = file_path.split('/')[:-1]
        if len(path_out) > 0:
            path = path_out[0]

            for i in range(len(path_out)):
                if not os.path.exists(path):
                    os.mkdir(path)

                if i < len(path_out) - 1:
                    path += '/' + path_out[i + 1]

    def get_file_list(self, root_dir,exclude="verify"):
        """
            get all files in root_dir directory
        """
        path_list = []
        file_list = []
        join_list = []
        for path, folder, files in os.walk(root_dir):
            if exclude in folder:
                folder.remove(exclude)
            for name in files:
                path_list.append(path)
                file_list.append(name)
                join_list.append(os.path.join(path, name))
        return path_list, file_list, join_list