import os

import cv2
from DataSetGenerator import DataSetGenerator
from opencv_utils import OpenCVHelper
from predictor import Predictor
from eye_detect import EyeBlinkDetector

eye_p = EyeBlinkDetector()
path = os.path.dirname(os.path.abspath(__file__))
class VerifyUser:
	def __init__(self):
		self.helper = OpenCVHelper()
		self.dg = DataSetGenerator(os.path.join(path,"training_face_rec"))
		MAX_LABELS  = len(self.dg.data_labels)

		self.predictor = Predictor(self.dg,num_labels=MAX_LABELS)
	def get_results(self,file_in):
		

		if not os.path.isfile(file_in):
			print("Captured file not found!")
			return False

		""" ------------------- Loading and process the video file ----------------- """
		num_blinks = eye_p.get_num_blinks(file_in)
		print "Num Blinks : %d " % num_blinks
		if num_blinks <=0:
			return None , "No Eye Blink found. !"
		cap = cv2.VideoCapture(file_in)
		results = list()
		while True:
			ret, image = cap.read()
			if not ret:
				break
			img_face, pos_face = self.helper.convert_img(image)
			if img_face is not None:
				cv2.rectangle(
					image, (pos_face[0], pos_face[2]), (pos_face[1], pos_face[3]), (255, 0, 0), 5)
				user , pred = self.predictor.classify_image(img_face)
				results.append(pred)
		if len(results) > 0:
			r = sum(results) / float(len(results))
			r = int(abs(r))
			return self.dg.data_labels[r] , True
		return None , "No Face Found !"
if __name__ == '__main__':
	v = VerifyUser()
	print v.get_results("media/verify/1.mp4")
