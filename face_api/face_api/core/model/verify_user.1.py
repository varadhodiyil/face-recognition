import os

import cv2
from DataSetGenerator import DataSetGenerator
from opencv_utils import OpenCVHelper
from predictor import Predictor
    
helper = OpenCVHelper()


def get_results(file_in):
	dg = DataSetGenerator("training_face_rec")
	MAX_LABELS  = len(dg.data_labels)
	p = Predictor(num_labels=MAX_LABELS)

	if not os.path.isfile(file_in):
		print("Captured file not found!")
		return False

	""" ------------------- Loading and process the video file ----------------- """
	cap = cv2.VideoCapture(file_in)
	results = list()
	while True:
		ret, image = cap.read()
		if not ret:
			break
		img_face, pos_face = helper.convert_img(image)
		if img_face is not None:
			cv2.rectangle(
				image, (pos_face[0], pos_face[2]), (pos_face[1], pos_face[3]), (255, 0, 0), 5)
			user , pred = p.classify_image(img_face)
			results.append(pred)
	if len(results) > 0:
		r = sum(results) / float(len(results))
		r = int(abs(r))
		print r
		return dg.data_labels[r]
	return None

get_results("/home/madhan/Documents/skalenow/face_api/face_api/core/model/media/madhan/1.mp4")